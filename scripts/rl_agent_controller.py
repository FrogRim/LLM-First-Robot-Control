#!/usr/bin/env python3
"""
RL 기반 로봇 제어 시스템
PPO (Proximal Policy Optimization) agent를 사용한 학습 기반 제어
"""

import json
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

try:
    import genesis as gs
    GENESIS_AVAILABLE = True
except ImportError:
    print("Warning: Genesis AI not installed. Simulation features will be disabled.")
    GENESIS_AVAILABLE = False


class PPOAgent(nn.Module):
    """간단한 PPO 에이전트"""

    def __init__(self, state_dim: int = 10, action_dim: int = 5, hidden_dim: int = 64):
        super(PPOAgent, self).__init__()

        # 정책 네트워크 (Actor)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # 가치 네트워크 (Critic)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.action_dim = action_dim

        # 디바이스 설정 (CPU로 고정)
        self.device = torch.device('cpu')
        self.to(self.device)

    def forward(self, state):
        """상태로부터 행동과 가치를 출력"""
        probs = self.actor(state)
        value = self.critic(state)
        return probs, value

    def get_action(self, state):
        """행동 선택"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            probs, value = self.forward(state_tensor)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()


class RLController:
    """RL 기반 제어 시스템 클래스"""

    def __init__(self, enable_genesis: bool = True):
        """
        Args:
            enable_genesis: Genesis AI 시뮬레이션 활성화
        """
        print("🤖 RL 기반 제어 시스템 초기화...")

        # 상태 및 행동 공간 정의
        self.state_dim = 10  # [물체 위치, 속도, 재료 타입 등]
        self.action_dim = 5  # [grip_force, lift_speed, approach_angle, contact_force, safety_margin] 범주화

        # PPO 에이전트 초기화
        self.agent = PPOAgent(self.state_dim, self.action_dim)

        # 간단한 사전 학습된 모델 로드 시도 (없으면 랜덤)
        self._load_pretrained_model()

        # Genesis AI 초기화
        self.genesis_enabled = enable_genesis and GENESIS_AVAILABLE
        self.genesis_scene = None
        self.scene_built = False

        if self.genesis_enabled:
            print("🌍 Genesis AI 초기화 중...")
            self._init_genesis()
            print("✓ Genesis AI 초기화 완료")
        else:
            print("⚠️  Genesis AI 비활성화 (시뮬레이션 없이 RL 추론만 실행)")

        # 재료 타입 매핑
        self.material_mapping = {
            'plastic': 0, 'metal': 1, 'glass': 2, 'wood': 3, 'rubber': 4
        }

        # 행동 매핑 (범주화된 값들)
        self.action_mapping = {
            0: {'grip_force': 0.3, 'lift_speed': 0.8, 'approach_angle': 0.0, 'contact_force': 0.1, 'safety_margin': 0.9},  # 조심스럽게
            1: {'grip_force': 0.5, 'lift_speed': 0.5, 'approach_angle': 0.0, 'contact_force': 0.3, 'safety_margin': 0.8},  # 표준
            2: {'grip_force': 0.7, 'lift_speed': 0.3, 'approach_angle': 0.0, 'contact_force': 0.5, 'safety_margin': 0.7},  # 세게
            3: {'grip_force': 0.4, 'lift_speed': 0.6, 'approach_angle': 0.1, 'contact_force': 0.2, 'safety_margin': 0.8},  # 측면 접근
            4: {'grip_force': 0.6, 'lift_speed': 0.4, 'approach_angle': -0.1, 'contact_force': 0.4, 'safety_margin': 0.6}   # 반대측 접근
        }

    def _load_pretrained_model(self):
        """사전 학습된 모델 로드 (없으면 랜덤 초기화 유지)"""
        try:
            # 간단한 사전 학습 데이터로 초기화 (실제로는 학습된 모델을 로드)
            print("📚 RL 모델 초기화 (랜덤 정책)")
            # 실제 프로젝트에서는 학습된 모델을 로드하는 코드 추가
        except:
            print("⚠️  사전 학습된 RL 모델을 찾을 수 없음 - 랜덤 정책 사용")

    def _init_genesis(self):
        """Genesis AI 엔진 초기화"""
        if not GENESIS_AVAILABLE:
            return

        # Genesis 초기화
        if not hasattr(gs, '_initialized') or not gs._initialized:
            gs.init(backend=gs.gpu)

        # 씬 생성
        self.genesis_scene = gs.Scene(
            show_viewer=False,
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
        self.genesis_scene.add_entity(gs.morphs.Plane())

        # 테이블 추가
        self.genesis_scene.add_entity(
            gs.morphs.Box(size=(1.0, 1.0, 0.05)),
            surface=gs.surfaces.Default(color=(0.8, 0.6, 0.4, 1.0)),
        )

    def _extract_state_from_command(self, command: str) -> np.ndarray:
        """명령어에서 상태 벡터 추출"""
        command_lower = command.lower()

        # 기본 상태 벡터 (10차원)
        state = np.zeros(self.state_dim)

        # 물체 위치/속도 (가정값 - 실제로는 센서에서 얻음)
        state[0:3] = [0.5, 0.0, 0.1]  # x, y, z 위치

        # 재료 타입 추론
        material_idx = 0  # default
        for material, idx in self.material_mapping.items():
            if material in command_lower:
                material_idx = idx
                break
        state[3] = material_idx / 4.0  # 정규화

        # 명령 타입 (pick/place/sort 등)
        if 'pick' in command_lower or 'grab' in command_lower:
            state[4] = 0.0
        elif 'place' in command_lower or 'put' in command_lower:
            state[4] = 0.5
        elif 'sort' in command_lower:
            state[4] = 1.0

        # 기타 상태 (랜덤 또는 기본값)
        state[5:10] = np.random.uniform(0, 1, 5)

        return state

    def _infer_material_from_command(self, command: str) -> str:
        """명령어에서 재료 추론"""
        command_lower = command.lower()

        material_keywords = {
            'plastic': ['plastic', 'bottle', 'container'],
            'metal': ['metal', 'steel', 'iron', 'tool'],
            'glass': ['glass', 'cup', 'fragile'],
            'wood': ['wood', 'block', 'board'],
            'rubber': ['rubber', 'ball']
        }

        for material, keywords in material_keywords.items():
            if any(keyword in command_lower for keyword in keywords):
                return material

        return 'plastic'  # 기본값

    def generate_physics_params(self, command: str) -> Dict[str, Any]:
        """
        명령어에서 물리 파라미터 생성 (RL 기반)

        Args:
            command: 자연어 명령

        Returns:
            물리 파라미터 JSON 딕셔너리
        """
        start_time = time.time()

        # 상태 추출
        state = self._extract_state_from_command(command)

        # RL 정책으로 행동 선택
        action_idx, log_prob, value = self.agent.get_action(state)

        # 행동을 제어 파라미터로 변환
        control_params = self.action_mapping[action_idx]

        # 재료 추론
        material = self._infer_material_from_command(command)

        # 물리 분석 생성 (RL 기반 추정)
        material_properties = {
            'plastic': {'mass': 'light', 'fragile': 'low', 'friction': 'medium'},
            'metal': {'mass': 'heavy', 'fragile': 'low', 'friction': 'high'},
            'glass': {'mass': 'medium', 'fragile': 'high', 'friction': 'low'},
            'wood': {'mass': 'medium', 'fragile': 'medium', 'friction': 'medium'},
            'rubber': {'mass': 'light', 'fragile': 'low', 'friction': 'high'}
        }

        props = material_properties.get(material, material_properties['plastic'])

        physical_analysis = {
            'material_inference': material,
            'mass_category': props['mass'],
            'friction_coefficient': props['friction'],
            'fragility': props['fragile'],
            'stiffness': 'medium',  # RL로는 추정하기 어려움
            'confidence': 0.7  # RL 기반은 약간 낮은 신뢰도
        }

        # 추론 시간 계산
        inference_time = (time.time() - start_time) * 1000

        reasoning = f"RL-based policy: Selected action {action_idx} for state vector. Estimated {material} material properties."

        # affordance assessment
        success_prob = 0.75 + np.random.uniform(-0.1, 0.1)  # RL 기반 약간 변동성
        risk_factors = ["Policy uncertainty", "Limited training data"]
        if props['fragile'] == 'high':
            risk_factors.append("High fragility detected")

        affordance_assessment = {
            'success_probability': max(0.5, min(0.95, success_prob)),
            'risk_factors': risk_factors,
            'recommended_approach': f"RL policy action {action_idx} for {material} handling"
        }

        result = {
            'physical_analysis': physical_analysis,
            'control_parameters': control_params,
            'reasoning': reasoning,
            'affordance_assessment': affordance_assessment,
            '_metadata': {
                'inference_time_ms': inference_time,
                'method': 'rl_based',
                'selected_action': action_idx,
                'state_vector': state.tolist(),
                'log_probability': log_prob,
                'estimated_value': value,
                'success': True
            }
        }

        return result

    def create_object_from_params(self, params: Dict[str, Any], object_name: str = "target_object") -> Optional[Any]:
        """물리 파라미터로 Genesis AI 객체 생성 (RL 버전)"""
        if not self.genesis_enabled or self.genesis_scene is None:
            return None

        if self.scene_built:
            return None

        try:
            physical = params.get('physical_analysis', {})
            material = physical.get('material_inference', 'plastic').lower()

            # 재료별 속성 매핑
            density_map = {
                'plastic': 1200.0, 'metal': 7800.0, 'wood': 600.0,
                'glass': 2500.0, 'rubber': 1500.0
            }
            rho = density_map.get(material, 1000.0)

            color_map = {
                'plastic': (0.2, 0.6, 1.0, 1.0),
                'metal': (0.7, 0.7, 0.7, 1.0),
                'glass': (0.6, 0.9, 1.0, 0.7),
                'wood': (0.6, 0.4, 0.2, 1.0),
                'rubber': (0.2, 0.2, 0.2, 1.0),
            }
            color = color_map.get(material, (0.5, 0.5, 0.5, 1.0))

            shape_preference = {
                'plastic': 'box', 'metal': 'box', 'wood': 'box',
                'glass': 'sphere', 'rubber': 'sphere'
            }
            shape_type = shape_preference.get(material, 'box')

            if shape_type == 'sphere':
                radius = 0.06
                morph = gs.morphs.Sphere(radius=radius)
            else:
                size = (0.08, 0.08, 0.15)
                morph = gs.morphs.Box(size=size)

            obj = self.genesis_scene.add_entity(
                morph,
                surface=gs.surfaces.Default(color=color),
            )

            print(f"✓ RL-Based: Genesis AI 객체 생성 ({material})")
            return obj

        except Exception as e:
            print(f"✗ RL-Based: 객체 생성 실패: {e}")
            return None

    def run_simulation(self, duration_sec: float = 2.0, real_time: bool = True) -> Dict[str, Any]:
        """Genesis AI 시뮬레이션 실행"""
        if not self.genesis_enabled:
            return {'success': False, 'error': 'Genesis AI not enabled'}

        try:
            if not self.scene_built:
                self.genesis_scene.build()
                self.scene_built = True

            dt = 0.01
            steps = int(duration_sec / dt)

            for step in range(steps):
                self.genesis_scene.step()
                if real_time and step % 2 == 0:
                    time.sleep(dt)

            return {
                'success': True,
                'duration_sec': duration_sec,
                'steps': steps
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def execute_command(self, command: str, simulate: bool = True, duration_sec: float = 2.0) -> Dict[str, Any]:
        """전체 파이프라인 실행: 명령어 → RL 정책 → 시뮬레이션"""
        print(f"\n{'='*70}")
        print(f"📝 RL 기반 명령: {command}")
        print(f"{'='*70}")

        # RL 정책 실행
        print("\n[1/3] RL 정책 실행 중...")
        start_time = time.time()
        params = self.generate_physics_params(command)
        policy_time = (time.time() - start_time) * 1000

        print(f"✓ RL 정책 실행 완료 ({policy_time:.1f}ms)")
        print("\n정책 결과:")
        print(json.dumps(params.get('physical_analysis', {}), indent=2, ensure_ascii=False))
        print(f"선택된 행동: {params['_metadata']['selected_action']}")

        # 시뮬레이션 실행
        if simulate and self.genesis_enabled:
            print("\n[2/3] Genesis AI 객체 생성 중...")
            obj = self.create_object_from_params(params, "rl_object")

            if obj is not None:
                print("\n[3/3] 시뮬레이션 실행 중...")
                sim_result = self.run_simulation(duration_sec=duration_sec)
            else:
                sim_result = {'success': False, 'error': 'Object creation failed'}
        else:
            sim_result = {'success': False, 'reason': 'Simulation disabled'}

        result = {
            'command': command,
            'success': True,
            'params': params,
            'policy_time_ms': policy_time,
            'simulation': sim_result
        }

        print(f"\n{'='*70}")
        print("✅ RL 기반 실행 완료")
        print(f"{'='*70}\n")

        return result

    def reset_scene(self):
        """씬 리셋"""
        if self.genesis_enabled and self.genesis_scene is not None:
            self.scene_built = False

    def cleanup(self):
        """리소스 정리"""
        pass


def main():
    """RL 기반 제어 시스템 테스트"""
    print("🧪 RL 기반 제어 시스템 테스트\n")

    controller = RLController(enable_genesis=True)

    test_commands = [
        "Pick up the plastic bottle gently",
        "Grab the heavy metal tool firmly",
        "Lift the glass cup very slowly",
        "Sort the rubber ball to the left side"
    ]

    try:
        for i, cmd in enumerate(test_commands, 1):
            print(f"\n\n{'#'*70}")
            print(f"테스트 케이스 {i}/{len(test_commands)}")
            print(f"{'#'*70}")

            result = controller.execute_command(cmd, simulate=True, duration_sec=2.0)

            if i < len(test_commands):
                controller.reset_scene()
                time.sleep(1)

    finally:
        controller.cleanup()


if __name__ == "__main__":
    main()
