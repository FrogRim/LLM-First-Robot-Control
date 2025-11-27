#!/usr/bin/env python3
"""
비교 실험 시나리오 정의 및 실행
3가지 주요 시나리오: Object Sorting, Multi-Step Rearrangement, Physical Property Reasoning
"""

import json
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

try:
    import genesis as gs
    GENESIS_AVAILABLE = True
except ImportError:
    print("Warning: Genesis AI not installed.")
    GENESIS_AVAILABLE = False


class ScenarioType(Enum):
    OBJECT_SORTING = "object_sorting"
    MULTI_STEP_REARRANGEMENT = "multi_step_rearrangement"
    PHYSICAL_PROPERTY_REASONING = "physical_property_reasoning"


class ExperimentScenario:
    """실험 시나리오 기본 클래스"""

    def __init__(self, scenario_type: ScenarioType, enable_genesis: bool = True):
        self.scenario_type = scenario_type
        self.enable_genesis = enable_genesis and GENESIS_AVAILABLE
        self.genesis_scene = None
        self.scene_built = False

        if self.enable_genesis:
            self._init_genesis_scene()

        # 시나리오별 설정
        self.setup_scenario()

    def _init_genesis_scene(self):
        """Genesis AI 씬 초기화"""
        if not GENESIS_AVAILABLE:
            return

        if not hasattr(gs, '_initialized') or not gs._initialized:
            gs.init(backend=gs.gpu)

        self.genesis_scene = gs.Scene(
            show_viewer=False,
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.0, 0.0, 2.0),
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

    def setup_scenario(self):
        """시나리오별 초기 설정"""
        raise NotImplementedError

    def get_test_commands(self) -> List[str]:
        """시나리오별 테스트 명령어들 반환"""
        raise NotImplementedError

    def evaluate_success(self, controller_result: Dict[str, Any]) -> Dict[str, Any]:
        """실행 결과 평가"""
        raise NotImplementedError

    def run_simulation(self, duration_sec: float = 1.0, real_time: bool = False) -> Dict[str, Any]:
        """시뮬레이션 실행"""
        if not self.enable_genesis or self.genesis_scene is None:
            return {'success': False, 'error': 'Genesis not enabled or scene not available'}

        try:
            # 씬이 빌드되지 않았다면 빌드
            if not self.scene_built:
                self.genesis_scene.build()
                self.scene_built = True

            # 시뮬레이션 실행
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

    def reset_scene(self):
        """씬 리셋"""
        self.scene_built = False
        if self.genesis_scene:
            # 새 씬 생성
            self._init_genesis_scene()
            self.setup_scenario()


class ObjectSortingScenario(ExperimentScenario):
    """Object Sorting Task: 다양한 재질의 물체를 규칙에 따라 정렬"""

    def setup_scenario(self):
        """정렬 작업을 위한 환경 설정"""
        if not self.genesis_scene:
            return

        # 테이블들 생성 (왼쪽: 고무/플라스틱, 오른쪽: 유리/금속)
        left_table = self.genesis_scene.add_entity(
            gs.morphs.Box(size=(0.3, 0.3, 0.05), pos=(-0.5, 0.0, 0.025)),
            surface=gs.surfaces.Default(color=(0.8, 0.8, 0.8, 1.0)),
        )

        right_table = self.genesis_scene.add_entity(
            gs.morphs.Box(size=(0.3, 0.3, 0.05), pos=(0.5, 0.0, 0.025)),
            surface=gs.surfaces.Default(color=(0.6, 0.6, 0.6, 1.0)),
        )

        # 중앙에 정렬할 물체들 배치
        objects = [
            ("plastic_bottle", (0.0, 0.2, 0.1), "plastic", "box"),
            ("metal_can", (0.15, 0.2, 0.1), "metal", "box"),
            ("glass_cup", (-0.15, 0.2, 0.08), "glass", "sphere"),
            ("rubber_ball", (0.0, -0.2, 0.06), "rubber", "sphere"),
        ]

        self.objects = []
        for name, pos, material, shape in objects:
            obj = self._create_object(name, pos, material, shape)
            self.objects.append((name, obj, material))

        self.target_zones = {
            'soft': (-0.5, 0.0),  # 왼쪽 테이블 (고무, 플라스틱)
            'hard': (0.5, 0.0),  # 오른쪽 테이블 (유리, 금속)
        }

    def _create_object(self, name: str, pos: Tuple, material: str, shape: str):
        """물체 생성 헬퍼"""
        color_map = {
            'plastic': (0.2, 0.6, 1.0, 1.0),
            'metal': (0.7, 0.7, 0.7, 1.0),
            'glass': (0.6, 0.9, 1.0, 0.7),
            'rubber': (0.2, 0.2, 0.2, 1.0),
        }

        if shape == 'sphere':
            morph = gs.morphs.Sphere(radius=0.04, pos=pos)
        else:
            morph = gs.morphs.Box(size=(0.06, 0.06, 0.08), pos=pos)

        return self.genesis_scene.add_entity(
            morph,
            surface=gs.surfaces.Default(color=color_map.get(material, (0.5, 0.5, 0.5, 1.0))),
        )

    def get_test_commands(self) -> List[str]:
        """정렬 작업 명령어들"""
        return [
            "Sort the plastic bottle to the left side",
            "Place the metal can on the right table",
            "Move the glass cup to the right side carefully",
            "Put the rubber ball on the left table",
            "Sort all soft materials to the left and hard materials to the right"
        ]

    def evaluate_success(self, controller_result: Dict[str, Any]) -> Dict[str, Any]:
        """정렬 성공 여부 평가"""
        command = controller_result.get('command', '').lower()
        params = controller_result.get('params', {})

        # 물체와 목표 위치 매핑
        object_target_map = {
            'plastic': 'soft', 'bottle': 'soft',
            'metal': 'hard', 'can': 'hard',
            'glass': 'hard', 'cup': 'hard',
            'rubber': 'hard', 'ball': 'soft'
        }

        # 명령어에서 물체와 목표 추론
        detected_object = None
        detected_target = None

        for obj, target in object_target_map.items():
            if obj in command:
                detected_object = obj
                detected_target = target
                break

        # Rule-Based와 RL의 정확도 평가 (단순화)
        material_analysis = params.get('physical_analysis', {})
        detected_material = material_analysis.get('material_inference', '').lower()

        # 성공 판정
        success = False
        reasoning = ""

        if detected_object and detected_target:
            # 재료 추론이 맞는지 확인
            if detected_material in object_target_map and object_target_map[detected_material] == detected_target:
                success = True
                reasoning = f"Correctly identified {detected_material} and assigned to {detected_target} zone"
            else:
                reasoning = f"Material detection mismatch: detected {detected_material}, expected target {detected_target}"
        else:
            reasoning = "Could not determine object or target from command"

        # 세부 메트릭
        metrics = {
            'success': success,
            'reasoning': reasoning,
            'detected_object': detected_object,
            'detected_target': detected_target,
            'detected_material': detected_material,
            'control_parameters': params.get('control_parameters', {}),
            'inference_time_ms': params.get('_metadata', {}).get('inference_time_ms', 0)
        }

        return metrics


class MultiStepRearrangementScenario(ExperimentScenario):
    """Multi-Step Rearrangement: 여러 물체를 목적지로 옮기는 작업"""

    def setup_scenario(self):
        """재배치 작업을 위한 환경 설정"""
        if not self.genesis_scene:
            return

        # 여러 테이블과 선반 생성
        table1 = self.genesis_scene.add_entity(
            gs.morphs.Box(size=(0.4, 0.4, 0.05), pos=(-0.6, 0.0, 0.025)),
            surface=gs.surfaces.Default(color=(0.8, 0.6, 0.4, 1.0)),
        )

        table2 = self.genesis_scene.add_entity(
            gs.morphs.Box(size=(0.4, 0.4, 0.05), pos=(0.6, 0.0, 0.025)),
            surface=gs.surfaces.Default(color=(0.6, 0.4, 0.8, 1.0)),
        )

        shelf = self.genesis_scene.add_entity(
            gs.morphs.Box(size=(0.3, 0.05, 0.6), pos=(0.0, 0.8, 0.3)),
            surface=gs.surfaces.Default(color=(0.4, 0.4, 0.4, 1.0)),
        )

        # 물체들 배치
        objects = [
            ("cup", (-0.6, 0.2, 0.1), "glass", "sphere"),
            ("book", (-0.4, 0.1, 0.08), "wood", "box"),
            ("pen", (-0.5, -0.1, 0.06), "plastic", "box"),
            ("plate", (0.6, 0.2, 0.08), "ceramic", "box"),
        ]

        self.objects = []
        for name, pos, material, shape in objects:
            obj = self._create_object(name, pos, material, shape)
            self.objects.append((name, obj, material))

        self.locations = {
            'kitchen_table': (-0.6, 0.0),
            'dining_table': (0.6, 0.0),
            'shelf': (0.0, 0.8),
        }

    def _create_object(self, name: str, pos: Tuple, material: str, shape: str):
        """물체 생성 헬퍼"""
        color_map = {
            'glass': (0.6, 0.9, 1.0, 0.7),
            'wood': (0.6, 0.4, 0.2, 1.0),
            'plastic': (0.2, 0.6, 1.0, 1.0),
            'ceramic': (0.9, 0.9, 0.9, 1.0),
        }

        if shape == 'sphere':
            morph = gs.morphs.Sphere(radius=0.04, pos=pos)
        else:
            morph = gs.morphs.Box(size=(0.06, 0.06, 0.08), pos=pos)

        return self.genesis_scene.add_entity(
            morph,
            surface=gs.surfaces.Default(color=color_map.get(material, (0.5, 0.5, 0.5, 1.0))),
        )

    def get_test_commands(self) -> List[str]:
        """다단계 재배치 명령어들"""
        return [
            "Move the cup from the kitchen table to the dining table",
            "Put the book on the shelf",
            "Place the pen next to the cup on the dining table",
            "Clear the kitchen table by moving everything to the shelf",
            "Rearrange items: cup and plate to dining table, book and pen to shelf"
        ]

    def evaluate_success(self, controller_result: Dict[str, Any]) -> Dict[str, Any]:
        """재배치 성공 여부 평가"""
        command = controller_result.get('command', '').lower()
        params = controller_result.get('params', {})

        # 명령어 분석
        object_location_map = {
            'cup': {'from': 'kitchen', 'to': 'dining' if 'dining' in command else None},
            'book': {'from': 'kitchen', 'to': 'shelf' if 'shelf' in command else None},
            'pen': {'from': 'kitchen', 'to': 'dining' if 'dining' in command else None},
            'plate': {'from': 'dining', 'to': 'dining' if 'dining' in command else None},
        }

        detected_objects = []
        target_locations = []

        for obj, mapping in object_location_map.items():
            if obj in command:
                detected_objects.append(obj)
                if mapping['to']:
                    target_locations.append(mapping['to'])

        # 계획 일관성 평가
        reasoning = params.get('reasoning', '')
        has_plan = 'plan' in reasoning.lower() or 'sequence' in reasoning.lower()

        # 성공 판정 (단순화)
        success = len(detected_objects) > 0 and has_plan

        metrics = {
            'success': success,
            'reasoning': f"Detected {len(detected_objects)} objects, plan consistency: {has_plan}",
            'detected_objects': detected_objects,
            'target_locations': target_locations,
            'has_plan': has_plan,
            'control_parameters': params.get('control_parameters', {}),
            'inference_time_ms': params.get('_metadata', {}).get('inference_time_ms', 0)
        }

        return metrics


class PhysicalPropertyReasoningScenario(ExperimentScenario):
    """Physical Property Reasoning Task: 물성 인식 + 전략 변화 요구"""

    def setup_scenario(self):
        """물성 추론 작업을 위한 환경 설정"""
        if not self.genesis_scene:
            return

        # 다양한 물체 배치
        objects = [
            ("fragile_glass", (0.0, 0.3, 0.08), "glass", "sphere"),
            ("heavy_metal", (0.2, 0.2, 0.12), "metal", "box"),
            ("light_plastic", (-0.2, 0.2, 0.08), "plastic", "box"),
            ("soft_rubber", (0.0, -0.2, 0.06), "rubber", "sphere"),
            ("wooden_block", (-0.2, -0.2, 0.1), "wood", "box"),
        ]

        self.objects = []
        for name, pos, material, shape in objects:
            obj = self._create_object(name, pos, material, shape)
            self.objects.append((name, obj, material))

        # 위험 구역 (떨어지면 안 되는 곳)
        hazard_zone = self.genesis_scene.add_entity(
            gs.morphs.Box(size=(0.2, 0.2, 0.02), pos=(0.0, -0.6, 0.01)),
            surface=gs.surfaces.Default(color=(1.0, 0.2, 0.2, 1.0)),  # 빨간색 위험 표시
        )

    def _create_object(self, name: str, pos: Tuple, material: str, shape: str):
        """물체 생성 헬퍼"""
        color_map = {
            'glass': (0.6, 0.9, 1.0, 0.7),
            'metal': (0.7, 0.7, 0.7, 1.0),
            'plastic': (0.2, 0.6, 1.0, 1.0),
            'rubber': (0.2, 0.2, 0.2, 1.0),
            'wood': (0.6, 0.4, 0.2, 1.0),
        }

        if shape == 'sphere':
            morph = gs.morphs.Sphere(radius=0.04, pos=pos)
        else:
            morph = gs.morphs.Box(size=(0.06, 0.06, 0.08), pos=pos)

        return self.genesis_scene.add_entity(
            morph,
            surface=gs.surfaces.Default(color=color_map.get(material, (0.5, 0.5, 0.5, 1.0))),
        )

    def get_test_commands(self) -> List[str]:
        """물성 추론 명령어들"""
        return [
            "Pick up the glass cup very carefully without breaking it",
            "Handle the heavy metal object with strong grip",
            "Move the light plastic item gently",
            "Be careful with the fragile glass - don't drop it",
            "The rubber ball is flexible, but the glass is not - handle them differently"
        ]

    def evaluate_success(self, controller_result: Dict[str, Any]) -> Dict[str, Any]:
        """물성 추론 성공 여부 평가"""
        command = controller_result.get('command', '').lower()
        params = controller_result.get('params', {})

        physical_analysis = params.get('physical_analysis', {})
        control_params = params.get('control_parameters', {})

        # 재료 추론 정확도
        material_inference = physical_analysis.get('material_inference', '').lower()
        fragility = physical_analysis.get('fragility', '')
        mass_category = physical_analysis.get('mass_category', '')

        # 명령어 기반 기대값
        expected_materials = []
        expected_fragility = []
        expected_mass = []

        if 'glass' in command or 'fragile' in command:
            expected_materials.append('glass')
            expected_fragility.append('high')
            expected_mass.append('medium')
        if 'metal' in command or 'heavy' in command:
            expected_materials.append('metal')
            expected_fragility.append('low')
            expected_mass.append('heavy')
        if 'plastic' in command or 'light' in command:
            expected_materials.append('plastic')
            expected_fragility.append('low')
            expected_mass.append('light')
        if 'rubber' in command:
            expected_materials.append('rubber')
            expected_fragility.append('low')
            expected_mass.append('light')

        # 정확도 평가
        material_correct = material_inference in expected_materials if expected_materials else True
        fragility_correct = fragility in expected_fragility if expected_fragility else True
        mass_correct = mass_category in expected_mass if expected_mass else True

        # 제어 파라미터 적절성 평가
        grip_force = control_params.get('grip_force', 0.5)
        lift_speed = control_params.get('lift_speed', 0.5)
        safety_margin = control_params.get('safety_margin', 0.8)

        # 유리/무거운 물체는 조심스러운 파라미터가 적절
        parameter_appropriate = True
        if 'glass' in command or 'fragile' in command:
            if grip_force > 0.5 or lift_speed > 0.4 or safety_margin < 0.8:
                parameter_appropriate = False
        elif 'metal' in command or 'heavy' in command:
            if grip_force < 0.6 or safety_margin > 0.7:
                parameter_appropriate = False

        # 종합 성공 판정
        success = material_correct and parameter_appropriate

        reasoning = f"Material: {material_correct}, Parameters: {parameter_appropriate}"

        metrics = {
            'success': success,
            'reasoning': reasoning,
            'material_correct': material_correct,
            'fragility_correct': fragility_correct,
            'mass_correct': mass_correct,
            'parameter_appropriate': parameter_appropriate,
            'detected_material': material_inference,
            'expected_materials': expected_materials,
            'control_parameters': control_params,
            'inference_time_ms': params.get('_metadata', {}).get('inference_time_ms', 0)
        }

        return metrics


def create_scenario(scenario_type: str, enable_genesis: bool = True) -> ExperimentScenario:
    """시나리오 타입에 따라 적절한 시나리오 객체 생성"""
    scenario_map = {
        'object_sorting': ObjectSortingScenario,
        'multi_step_rearrangement': MultiStepRearrangementScenario,
        'physical_property_reasoning': PhysicalPropertyReasoningScenario,
    }

    scenario_class = scenario_map.get(scenario_type.lower())
    if not scenario_class:
        raise ValueError(f"Unknown scenario type: {scenario_type}")

    scenario_enum = ScenarioType(scenario_type.lower())
    return scenario_class(scenario_enum, enable_genesis)


def run_scenario_test(scenario: ExperimentScenario, controller_class, controller_name: str,
                     num_runs: int = 3) -> Dict[str, Any]:
    """특정 시나리오에 대해 컨트롤러 테스트 실행"""
    print(f"\n🧪 {controller_name} - {scenario.scenario_type.value} 시나리오 테스트")
    print("="*80)

    # 컨트롤러 초기화 (씬 공유를 위해 Genesis 완전 비활성화)
    controller = controller_class(enable_genesis=False)

    # 컨트롤러의 Genesis 관련 속성 강제 설정
    if hasattr(controller, 'genesis_enabled'):
        controller.genesis_enabled = False
    if hasattr(controller, 'llm_integration') and hasattr(controller.llm_integration, 'genesis_enabled'):
        controller.llm_integration.genesis_enabled = False

    results = []
    commands = scenario.get_test_commands()

    try:
        for run_idx in range(min(num_runs, len(commands))):
            command = commands[run_idx]

            print(f"\n[Test {run_idx + 1}/{num_runs}] {command}")

            # 컨트롤러 실행 (시뮬레이션 없이 파라미터 생성만)
            controller_start = time.time()
            controller_result = controller.execute_command(command, simulate=False)
            controller_time = time.time() - controller_start

            # 시나리오에서 시뮬레이션 실행 (공유된 씬 사용)
            sim_start = time.time()
            if scenario.enable_genesis and scenario.genesis_scene:
                # 시뮬레이션 실행 (간단한 물리 시뮬레이션)
                scenario.run_simulation(duration_sec=1.0, real_time=False)
                sim_success = True
            else:
                sim_success = False
            sim_time = time.time() - sim_start

            total_time = controller_time + sim_time

            # 평가 (컨트롤러 결과 + 시뮬레이션 성공 여부)
            evaluation = scenario.evaluate_success(controller_result)
            evaluation['simulation_success'] = sim_success
            evaluation['controller_time_sec'] = controller_time
            evaluation['simulation_time_sec'] = sim_time

            # 결과 저장
            result = {
                'run_idx': run_idx,
                'command': command,
                'controller_result': controller_result,
                'evaluation': evaluation,
                'total_time_sec': total_time,
                'success': evaluation.get('success', False)
            }
            results.append(result)

            # 씬 리셋
            if run_idx < num_runs - 1:
                scenario.reset_scene()
                controller.reset_scene()
                time.sleep(0.5)

    finally:
        controller.cleanup()

    # 요약 통계
    success_count = sum(1 for r in results if r['success'])
    success_rate = success_count / len(results) if results else 0
    avg_time = np.mean([r['total_time_sec'] for r in results]) if results else 0
    avg_inference_time = np.mean([r['evaluation'].get('inference_time_ms', 0) for r in results]) if results else 0

    summary = {
        'scenario_type': scenario.scenario_type.value,
        'controller_name': controller_name,
        'total_runs': len(results),
        'success_count': success_count,
        'success_rate': success_rate,
        'average_total_time_sec': avg_time,
        'average_inference_time_ms': avg_inference_time,
        'detailed_results': results
    }

    print("\n📊 테스트 요약:")
    print(f"- 총 실행 수: {len(results)}")
    print(f"- 성공률: {success_rate:.1%}")
    print(f"- 평균 추론 시간: {avg_inference_time:.1f}ms")
    return summary


if __name__ == "__main__":
    """시나리오 테스트"""
    print("🧪 실험 시나리오 테스트\n")

    # Object Sorting 시나리오 테스트
    sorting_scenario = create_scenario('object_sorting', enable_genesis=True)

    # 간단한 테스트 (실제 비교 실험에서는 각 컨트롤러별로 실행)
    print("Object Sorting 시나리오 명령어들:")
    for i, cmd in enumerate(sorting_scenario.get_test_commands(), 1):
        print(f"{i}. {cmd}")
