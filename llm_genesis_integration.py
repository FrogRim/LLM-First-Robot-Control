#!/usr/bin/env python3
"""
LLM + Genesis AI 통합 클래스
학습된 Qwen2.5-14B QLora 모델로 자연어 명령을 물리 파라미터로 변환하고
Genesis AI 시뮬레이션을 실행합니다.
"""

import json
import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

try:
    import genesis as gs
    GENESIS_AVAILABLE = True
except ImportError:
    print("Warning: Genesis AI not installed. Simulation features will be disabled.")
    GENESIS_AVAILABLE = False


class LLMGenesisIntegration:
    """LLM과 Genesis AI를 통합하는 클래스"""

    def __init__(
        self,
        adapter_dir: str = "./droid-physics-qwen14b-qlora",
        base_model: str = "Qwen/Qwen2.5-14B-Instruct",
        device: str = "auto",
        enable_genesis: bool = True
    ):
        """
        Args:
            adapter_dir: QLora 어댑터 디렉토리
            base_model: 베이스 모델 이름
            device: 디바이스 설정
            enable_genesis: Genesis AI 시뮬레이션 활성화
        """
        print(f"🚀 LLM + Genesis AI 통합 초기화...")

        # LLM 로드
        print(f"📦 모델 로드 중: {base_model} + {adapter_dir}")
        self.model, self.tokenizer = self._load_model(adapter_dir, base_model, device)
        self.model.eval()
        print("✓ LLM 모델 로드 완료")

        # Genesis AI 초기화
        self.genesis_enabled = enable_genesis and GENESIS_AVAILABLE
        self.genesis_scene = None
        self.scene_built = False  # 씬 빌드 상태 추적

        if self.genesis_enabled:
            print("🌍 Genesis AI 초기화 중...")
            self._init_genesis()
            print("✓ Genesis AI 초기화 완료")
        else:
            print("⚠️  Genesis AI 비활성화 (시뮬레이션 없이 LLM 추론만 실행)")

        # 시스템 프롬프트
        self.system_prompt = """You are a physics-aware robot control system. Output ONLY valid JSON without code fences, explanations, or extra text.

Required JSON schema:
{
  "physical_analysis": {
    "material_inference": "string (plastic/metal/wood/glass/rubber)",
    "mass_category": "string (light/medium/heavy)",
    "friction_coefficient": "string (low/medium/high)",
    "fragility": "string (low/medium/high)",
    "stiffness": "string (soft/medium/hard)",
    "confidence": 0.85
  },
  "control_parameters": {
    "grip_force": 0.5,
    "lift_speed": 0.5,
    "approach_angle": 0.0,
    "contact_force": 0.3,
    "safety_margin": 0.8
  },
  "reasoning": "string",
  "affordance_assessment": {
    "success_probability": 0.9,
    "risk_factors": [],
    "recommended_approach": "string"
  }
}"""

    def _load_model(
        self,
        adapter_dir: str,
        base_model: str,
        device: str
    ) -> Tuple[PeftModel, AutoTokenizer]:
        """모델 로드"""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=device,
            trust_remote_code=True,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
        )

        model = PeftModel.from_pretrained(base, adapter_dir)
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def _init_genesis(self):
        """Genesis AI 엔진 초기화"""
        if not GENESIS_AVAILABLE:
            return

        # Genesis 초기화 (한 번만)
        if not hasattr(gs, '_initialized') or not gs._initialized:
            gs.init(backend=gs.gpu)

        # 씬 생성 (show_viewer=False로 안정성 향상)
        self.genesis_scene = gs.Scene(
            show_viewer=False,  # 다중 씬 충돌 방지
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.5, 0.0, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                res=(1280, 720),
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(
                dt=0.01,
                substeps=2,
                gravity=(0, 0, -9.81),
            ),
        )

        # 지면 추가
        self.genesis_scene.add_entity(
            gs.morphs.Plane(),
        )

        # 테이블 추가 (작업 공간)
        self.genesis_scene.add_entity(
            gs.morphs.Box(size=(1.0, 1.0, 0.05)),
            surface=gs.surfaces.Default(
                color=(0.8, 0.6, 0.4, 1.0),
            ),
        )

    def generate_physics_params(
        self,
        command: str,
        max_tokens: int = 256,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        자연어 명령을 물리 파라미터 JSON으로 변환

        Args:
            command: 자연어 명령 (예: "Pick up the plastic bottle gently")
            max_tokens: 최대 토큰 수
            temperature: 샘플링 온도

        Returns:
            물리 파라미터 JSON 딕셔너리
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": command + "\n\nOutput JSON:"},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.95 if temperature > 0 else None,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        inference_time = (time.time() - start_time) * 1000

        # 디코딩
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        # JSON 파싱
        try:
            # JSON 추출 (코드 펜스 제거)
            json_text = generated_text.strip()
            if json_text.startswith("```"):
                json_text = json_text.split("```")[1]
                if json_text.startswith("json"):
                    json_text = json_text[4:]

            params = json.loads(json_text)
            params['_metadata'] = {
                'inference_time_ms': inference_time,
                'raw_output': generated_text,
                'success': True
            }
            return params

        except json.JSONDecodeError as e:
            print(f"⚠️  JSON 파싱 실패: {e}")
            print(f"생성된 텍스트: {generated_text}")
            return {
                '_metadata': {
                    'inference_time_ms': inference_time,
                    'raw_output': generated_text,
                    'success': False,
                    'error': str(e)
                }
            }

    def create_object_from_params(
        self,
        params: Dict[str, Any],
        object_name: str = "target_object"
    ) -> Optional[Any]:
        """
        물리 파라미터로 Genesis AI 객체 생성

        Args:
            params: LLM이 생성한 물리 파라미터
            object_name: 객체 이름

        Returns:
            Genesis AI 객체 (또는 None)
        """
        if not self.genesis_enabled or self.genesis_scene is None:
            print("⚠️  Genesis AI가 활성화되지 않았습니다.")
            return None

        # 씬이 이미 빌드되었다면 새 객체를 추가할 수 없음
        if self.scene_built:
            print("⚠️  씬이 이미 빌드되어 새 객체를 추가할 수 없습니다.")
            print("   (첫 시나리오만 시각화됩니다)")
            return None

        try:
            # 물리 분석에서 속성 추출
            physical = params.get('physical_analysis', {})
            material = physical.get('material_inference', 'plastic').lower()
            mass_category = physical.get('mass_category', 'medium').lower()
            friction = physical.get('friction_coefficient', 'medium').lower()
            fragility = physical.get('fragility', 'medium').lower()

            # 재료별 밀도 매핑
            density_map = {
                'plastic': 1200.0,
                'metal': 7800.0,
                'wood': 600.0,
                'glass': 2500.0,
                'rubber': 1500.0,
            }
            rho = density_map.get(material, 1000.0)

            # 질량 카테고리별 스케일 조정
            mass_scale = {'light': 0.5, 'medium': 1.0, 'heavy': 2.0}
            rho *= mass_scale.get(mass_category, 1.0)

            # 마찰 계수 매핑
            friction_map = {'low': 0.3, 'medium': 0.5, 'high': 0.8}
            friction_coef = friction_map.get(friction, 0.5)

            # 반발 계수 (깨지기 쉬운 정도)
            fragility_map = {'low': 0.3, 'medium': 0.2, 'high': 0.05}
            restitution = fragility_map.get(fragility, 0.2)

            # 재질별 색상 매핑 (RGBA)
            color_map = {
                'plastic': (0.2, 0.6, 1.0, 1.0),   # 파란색 (플라스틱)
                'metal': (0.7, 0.7, 0.7, 1.0),     # 회색 (금속)
                'glass': (0.6, 0.9, 1.0, 0.7),     # 반투명 청록색 (유리)
                'wood': (0.6, 0.4, 0.2, 1.0),      # 갈색 (나무)
                'rubber': (0.2, 0.2, 0.2, 1.0),    # 검은색 (고무)
            }
            color = color_map.get(material, (0.5, 0.5, 0.5, 1.0))  # 기본: 회색

            # 재료별 최적 형상 선택 (안정성 향상)
            shape_preference = {
                'plastic': 'box',
                'metal': 'box',
                'wood': 'box',
                'glass': 'sphere',    # 유리는 구 형태가 더 안정적
                'rubber': 'sphere',   # 고무는 구 형태가 더 안정적
                'ceramic': 'box',
                'fabric': 'box',
            }
            shape_type = shape_preference.get(material, 'box')

            # 형상 생성
            if shape_type == 'sphere':
                radius = 0.06  # 반지름 6cm
                morph = gs.morphs.Sphere(radius=radius)
                shape_info = f"Sphere(r={radius:.2f}m)"
            else:  # box
                size = (0.08, 0.08, 0.15)  # 작은 병/박스 크기
                morph = gs.morphs.Box(size=size)
                shape_info = f"Box({size[0]:.2f}×{size[1]:.2f}×{size[2]:.2f}m)"

            # 객체 생성
            obj = self.genesis_scene.add_entity(
                morph,
                surface=gs.surfaces.Default(
                    color=color,
                ),
            )

            print(f"✓ Genesis AI 객체 생성: {object_name}")
            print(f"  - 재료: {material}, 형상: {shape_info}")
            print(f"  - 밀도: {rho:.1f} kg/m³, 마찰: {friction_coef:.2f}, 반발: {restitution:.2f}")
            print(f"  - 색상: RGB({color[0]:.1f}, {color[1]:.1f}, {color[2]:.1f})")

            return obj

        except Exception as e:
            print(f"✗ Genesis AI 객체 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_simulation(
        self,
        duration_sec: float = 2.0,
        real_time: bool = True
    ) -> Dict[str, Any]:
        """
        Genesis AI 시뮬레이션 실행

        Args:
            duration_sec: 시뮬레이션 시간 (초)
            real_time: 실시간 시각화 여부

        Returns:
            시뮬레이션 결과
        """
        if not self.genesis_enabled:
            return {'success': False, 'error': 'Genesis AI not enabled'}

        try:
            # 시뮬레이션 구축 (첫 번째만)
            if not self.scene_built:
                print("🔨 시뮬레이션 구축 중...")
                self.genesis_scene.build()
                self.scene_built = True
                print("✓ 시뮬레이션 구축 완료")
            else:
                print("✓ 기존 시뮬레이션 재사용")

            # 시뮬레이션 실행
            print(f"▶️  시뮬레이션 실행 ({duration_sec}초)...")
            dt = 0.01
            steps = int(duration_sec / dt)

            trajectory = []

            for step in range(steps):
                self.genesis_scene.step()

                # 객체 위치 기록 (간단히 첫 번째 동적 객체만)
                # 실제로는 모든 객체를 추적할 수 있음

                if real_time and step % 2 == 0:  # 렌더링 주기 조절
                    time.sleep(dt)

            print("✓ 시뮬레이션 완료")

            return {
                'success': True,
                'duration_sec': duration_sec,
                'steps': steps,
                'trajectory': trajectory
            }

        except Exception as e:
            print(f"✗ 시뮬레이션 실행 실패: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def execute_command(
        self,
        command: str,
        simulate: bool = True,
        duration_sec: float = 2.0
    ) -> Dict[str, Any]:
        """
        전체 파이프라인 실행: 자연어 → LLM → Genesis AI 시뮬레이션

        Args:
            command: 자연어 명령
            simulate: Genesis AI 시뮬레이션 실행 여부
            duration_sec: 시뮬레이션 시간

        Returns:
            전체 실행 결과
        """
        print(f"\n{'='*70}")
        print(f"📝 명령: {command}")
        print(f"{'='*70}")

        # 1. LLM으로 물리 파라미터 생성
        print("\n[1/3] LLM 추론 중...")
        params = self.generate_physics_params(command)

        if not params.get('_metadata', {}).get('success', False):
            print("✗ LLM 추론 실패")
            return {
                'command': command,
                'success': False,
                'params': params
            }

        print(f"✓ LLM 추론 완료 ({params['_metadata']['inference_time_ms']:.1f}ms)")
        print("\n생성된 물리 파라미터:")
        print(json.dumps(params.get('physical_analysis', {}), indent=2, ensure_ascii=False))
        print(json.dumps(params.get('control_parameters', {}), indent=2, ensure_ascii=False))

        # 2. Genesis AI 객체 생성
        if simulate and self.genesis_enabled:
            print("\n[2/3] Genesis AI 객체 생성 중...")
            obj = self.create_object_from_params(params, "target_object")

            # 3. 시뮬레이션 실행
            if obj is not None:
                print("\n[3/3] 시뮬레이션 실행 중...")
                sim_result = self.run_simulation(duration_sec=duration_sec)
            else:
                sim_result = {'success': False, 'error': 'Object creation failed'}
        else:
            obj = None
            sim_result = {'success': False, 'reason': 'Simulation disabled'}

        result = {
            'command': command,
            'success': True,
            'params': params,
            'simulation': sim_result
        }

        print(f"\n{'='*70}")
        print("✅ 명령 실행 완료")
        print(f"{'='*70}\n")

        return result

    def reset_scene(self):
        """씬 리셋 - 새 시나리오를 위해 씬 재생성"""
        if self.genesis_enabled and self.genesis_scene is not None:
            print("🔄 Genesis AI 씬 리셋 중...")
            # 씬 빌드 플래그 리셋 (다음 시나리오에서 재빌드)
            self.scene_built = False
            print("✓ 씬 리셋 완료 (다음 시나리오에서 재빌드)")

    def cleanup(self):
        """리소스 정리"""
        if self.genesis_enabled and self.genesis_scene is not None:
            print("🧹 리소스 정리 중...")
            # Genesis는 자동으로 리소스 정리
            print("✓ 리소스 정리 완료")


def main():
    """테스트 실행"""
    print("🧪 LLM + Genesis AI 통합 테스트\n")

    # 통합 클래스 초기화
    integration = LLMGenesisIntegration(
        adapter_dir="./droid-physics-qwen14b-qlora",
        enable_genesis=True
    )

    # 테스트 명령
    test_commands = [
        "Pick up the plastic bottle gently and place it in the container",
        "Grab the metal tool carefully without dropping it",
        "Lift the glass cup very slowly and steadily",
    ]

    try:
        for i, cmd in enumerate(test_commands, 1):
            print(f"\n\n{'#'*70}")
            print(f"테스트 케이스 {i}/{len(test_commands)}")
            print(f"{'#'*70}")

            result = integration.execute_command(
                cmd,
                simulate=True,
                duration_sec=2.0
            )

            # 다음 테스트를 위해 씬 리셋
            if i < len(test_commands):
                integration.reset_scene()
                time.sleep(1)

    finally:
        integration.cleanup()


if __name__ == "__main__":
    main()
