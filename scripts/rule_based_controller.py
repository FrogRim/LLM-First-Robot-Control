#!/usr/bin/env python3
"""
Rule-Based 로봇 제어 시스템
전통적 if-else 기반 조건 분기 방식으로 물체를 인식하고 조작
"""

import json
import time
from typing import Dict, Any, List, Optional
import numpy as np

try:
    import genesis as gs
    GENESIS_AVAILABLE = True
except ImportError:
    print("Warning: Genesis AI not installed. Simulation features will be disabled.")
    GENESIS_AVAILABLE = False


class RuleBasedController:
    """Rule-Based 제어 시스템 클래스"""

    def __init__(self, enable_genesis: bool = True):
        """
        Args:
            enable_genesis: Genesis AI 시뮬레이션 활성화
        """
        print("🔧 Rule-Based 제어 시스템 초기화...")

        # Genesis AI 초기화
        self.genesis_enabled = enable_genesis and GENESIS_AVAILABLE
        self.genesis_scene = None
        self.scene_built = False

        if self.genesis_enabled:
            print("🌍 Genesis AI 초기화 중...")
            self._init_genesis()
            print("✓ Genesis AI 초기화 완료")
        else:
            print("⚠️  Genesis AI 비활성화 (시뮬레이션 없이 룰 추론만 실행)")

        # 룰 베이스 정의
        self.material_rules = self._define_material_rules()
        self.action_rules = self._define_action_rules()

    def _define_material_rules(self) -> Dict[str, Dict[str, Any]]:
        """재료별 물리 속성 룰 정의"""
        return {
            # 재료 키워드 -> 물리 속성 매핑
            'plastic': {
                'material_inference': 'plastic',
                'mass_category': 'light',
                'friction_coefficient': 'medium',
                'fragility': 'low',
                'stiffness': 'medium',
                'grip_force': 0.4,
                'lift_speed': 0.6,
                'approach_angle': 0.0,
                'contact_force': 0.2,
                'safety_margin': 0.7
            },
            'metal': {
                'material_inference': 'metal',
                'mass_category': 'heavy',
                'friction_coefficient': 'high',
                'fragility': 'low',
                'stiffness': 'hard',
                'grip_force': 0.8,
                'lift_speed': 0.3,
                'approach_angle': 0.0,
                'contact_force': 0.4,
                'safety_margin': 0.8
            },
            'glass': {
                'material_inference': 'glass',
                'mass_category': 'medium',
                'friction_coefficient': 'low',
                'fragility': 'high',
                'stiffness': 'hard',
                'grip_force': 0.3,
                'lift_speed': 0.2,
                'approach_angle': 0.0,
                'contact_force': 0.1,
                'safety_margin': 0.9
            },
            'wood': {
                'material_inference': 'wood',
                'mass_category': 'medium',
                'friction_coefficient': 'medium',
                'fragility': 'medium',
                'stiffness': 'medium',
                'grip_force': 0.5,
                'lift_speed': 0.4,
                'approach_angle': 0.0,
                'contact_force': 0.3,
                'safety_margin': 0.8
            },
            'rubber': {
                'material_inference': 'rubber',
                'mass_category': 'light',
                'friction_coefficient': 'high',
                'fragility': 'low',
                'stiffness': 'soft',
                'grip_force': 0.6,
                'lift_speed': 0.7,
                'approach_angle': 0.0,
                'contact_force': 0.3,
                'safety_margin': 0.6
            },
            # 기본값
            'default': {
                'material_inference': 'plastic',
                'mass_category': 'medium',
                'friction_coefficient': 'medium',
                'fragility': 'medium',
                'stiffness': 'medium',
                'grip_force': 0.5,
                'lift_speed': 0.5,
                'approach_angle': 0.0,
                'contact_force': 0.3,
                'safety_margin': 0.8
            }
        }

    def _define_action_rules(self) -> Dict[str, Dict[str, Any]]:
        """행동별 룰 정의"""
        return {
            'pick': {
                'sequence': ['approach', 'grip', 'lift', 'hold'],
                'time_per_action': 0.5,
                'success_probability': 0.85
            },
            'place': {
                'sequence': ['move', 'lower', 'release'],
                'time_per_action': 0.3,
                'success_probability': 0.90
            },
            'sort': {
                'sequence': ['identify', 'pick', 'move', 'place'],
                'time_per_action': 0.4,
                'success_probability': 0.80
            },
            'default': {
                'sequence': ['approach', 'grip', 'lift'],
                'time_per_action': 0.5,
                'success_probability': 0.75
            }
        }

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

    def _infer_material_from_command(self, command: str) -> str:
        """명령어에서 재료 추론"""
        command_lower = command.lower()

        # 키워드 기반 재료 추론
        material_keywords = {
            'plastic': ['plastic', 'bottle', 'container', 'cup'],
            'metal': ['metal', 'steel', 'iron', 'tool', 'can'],
            'glass': ['glass', 'cup', 'bottle', 'fragile'],
            'wood': ['wood', 'block', 'board', 'table'],
            'rubber': ['rubber', 'ball', 'elastic']
        }

        for material, keywords in material_keywords.items():
            if any(keyword in command_lower for keyword in keywords):
                return material

        return 'default'

    def _infer_action_from_command(self, command: str) -> str:
        """명령어에서 행동 추론"""
        command_lower = command.lower()

        if 'sort' in command_lower or 'separate' in command_lower:
            return 'sort'
        elif 'place' in command_lower or 'put' in command_lower:
            return 'place'
        elif 'pick' in command_lower or 'grab' in command_lower:
            return 'pick'

        return 'default'

    def generate_physics_params(self, command: str) -> Dict[str, Any]:
        """
        명령어에서 물리 파라미터 생성 (Rule-Based)

        Args:
            command: 자연어 명령

        Returns:
            물리 파라미터 JSON 딕셔너리
        """
        start_time = time.time()

        # 재료와 행동 추론
        material = self._infer_material_from_command(command)
        action = self._infer_action_from_command(command)

        # 룰 기반 속성 가져오기
        material_props = self.material_rules[material]
        action_props = self.action_rules[action]

        # 물리 분석 생성
        physical_analysis = {
            'material_inference': material_props['material_inference'],
            'mass_category': material_props['mass_category'],
            'friction_coefficient': material_props['friction_coefficient'],
            'fragility': material_props['fragility'],
            'stiffness': material_props['stiffness'],
            'confidence': 0.8  # Rule-Based는 고정 신뢰도
        }

        # 제어 파라미터 생성
        control_parameters = {
            'grip_force': material_props['grip_force'],
            'lift_speed': material_props['lift_speed'],
            'approach_angle': material_props['approach_angle'],
            'contact_force': material_props['contact_force'],
            'safety_margin': material_props['safety_margin']
        }

        # 추론 시간 계산
        inference_time = (time.time() - start_time) * 1000

        # reasoning 생성
        reasoning = f"Rule-based analysis: Detected {material} material and {action} action. Applied predefined parameters."

        # affordance assessment
        success_prob = action_props['success_probability']
        risk_factors = []
        if material_props['fragility'] == 'high':
            risk_factors.append("High fragility - careful handling required")
        if material_props['mass_category'] == 'heavy':
            risk_factors.append("Heavy object - increased grip force needed")

        affordance_assessment = {
            'success_probability': success_prob,
            'risk_factors': risk_factors,
            'recommended_approach': f"Standard {action} procedure for {material}"
        }

        # 결과 조합
        result = {
            'physical_analysis': physical_analysis,
            'control_parameters': control_parameters,
            'reasoning': reasoning,
            'affordance_assessment': affordance_assessment,
            '_metadata': {
                'inference_time_ms': inference_time,
                'method': 'rule_based',
                'detected_material': material,
                'detected_action': action,
                'success': True
            }
        }

        return result

    def create_object_from_params(self, params: Dict[str, Any], object_name: str = "target_object") -> Optional[Any]:
        """물리 파라미터로 Genesis AI 객체 생성 (Rule-Based 버전)"""
        if not self.genesis_enabled or self.genesis_scene is None:
            return None

        if self.scene_built:
            return None

        try:
            # 물리 분석에서 속성 추출
            physical = params.get('physical_analysis', {})
            material = physical.get('material_inference', 'plastic').lower()

            # 재료별 속성 매핑 (LLM 버전과 동일하게)
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

            # 형상 선택
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

            # 객체 생성
            obj = self.genesis_scene.add_entity(
                morph,
                surface=gs.surfaces.Default(color=color),
            )

            print(f"✓ Rule-Based: Genesis AI 객체 생성 ({material})")
            return obj

        except Exception as e:
            print(f"✗ Rule-Based: 객체 생성 실패: {e}")
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
        """전체 파이프라인 실행: 명령어 → Rule-Based 분석 → 시뮬레이션"""
        print(f"\n{'='*70}")
        print(f"📝 Rule-Based 명령: {command}")
        print(f"{'='*70}")

        # Rule-Based 분석
        print("\n[1/3] Rule-Based 분석 중...")
        start_time = time.time()
        params = self.generate_physics_params(command)
        analysis_time = (time.time() - start_time) * 1000

        print(f"✓ Rule-Based 분석 완료 ({analysis_time:.1f}ms)")
        print("\n분석 결과:")
        print(json.dumps(params.get('physical_analysis', {}), indent=2, ensure_ascii=False))

        # 시뮬레이션 실행
        if simulate and self.genesis_enabled:
            print("\n[2/3] Genesis AI 객체 생성 중...")
            obj = self.create_object_from_params(params, "rule_based_object")

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
            'analysis_time_ms': analysis_time,
            'simulation': sim_result
        }

        print(f"\n{'='*70}")
        print("✅ Rule-Based 실행 완료")
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
    """Rule-Based 제어 시스템 테스트"""
    print("🧪 Rule-Based 제어 시스템 테스트\n")

    controller = RuleBasedController(enable_genesis=True)

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
