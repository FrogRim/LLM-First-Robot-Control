#!/usr/bin/env python3
"""
LLM-First 자연어 파싱 계층 (LLM-First Natural Language Parsing Layer)

졸업논문: "LLM-First 기반 물리 속성 추출 로봇 제어"
목표: 자연어 명령 → 물리속성 + 제어파라미터 추출

예시: "무거운 금속 상자를 선반에 올려놔"
     → {"질량": "heavy", "재질": "metal", "그립력": "높게", "이동속도": "느리게"}
"""

import json
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from datetime import datetime

# 기존 모듈 연동
from data_abstraction_layer import PhysicalProperties, RobotConfiguration
from physics_mapping_layer import UniversalPhysicsMapper

logger = logging.getLogger(__name__)

# ============================================================================
# 물리 속성 및 제어 파라미터 데이터 구조
# ============================================================================

class PhysicalAttribute(Enum):
    """물리 속성 분류"""
    MASS = "mass"                    # 질량: heavy, light, medium
    FRICTION = "friction"            # 마찰: slippery, rough, smooth
    STIFFNESS = "stiffness"         # 강성: soft, hard, flexible
    FRAGILITY = "fragility"         # 깨지기 쉬움: fragile, sturdy, robust
    TEMPERATURE = "temperature"      # 온도: hot, cold, warm
    TEXTURE = "texture"             # 질감: smooth, rough, bumpy

class ActionIntent(Enum):
    """로봇 동작 의도"""
    PICK = "pick"                   # 집기
    PLACE = "place"                 # 놓기
    MOVE = "move"                   # 이동
    PUSH = "push"                   # 밀기
    PULL = "pull"                   # 당기기
    ROTATE = "rotate"               # 회전
    INSPECT = "inspect"             # 검사
    STOP = "stop"                   # 정지

@dataclass
class ExtractedPhysicalProperties:
    """추출된 물리 속성"""
    mass: str = "medium"            # heavy, medium, light
    friction: str = "normal"        # high, normal, low
    stiffness: str = "medium"       # hard, medium, soft
    fragility: str = "normal"       # fragile, normal, robust
    material: str = "unknown"       # metal, plastic, wood, glass, etc.
    temperature: str = "ambient"    # hot, warm, ambient, cool, cold
    confidence: float = 0.0         # 추출 신뢰도 [0-1]

@dataclass
class ControlParameters:
    """로봇 제어 파라미터"""
    grip_force: float = 0.5         # 그립력 [0-1]
    lift_speed: float = 0.5         # 들어올리기 속도 [0-1]
    approach_angle: float = 0.0     # 접근 각도 [도]
    contact_force: float = 0.3      # 접촉력 [0-1]
    safety_margin: float = 0.8      # 안전 여유도 [0-1]
    execution_time: float = 2.0     # 예상 실행 시간 [초]

@dataclass
class AffordanceAssessment:
    """어포던스 평가 결과"""
    affordances: List[str]          # 가능한 동작들
    success_probability: float      # 성공 확률 [0-1]
    risk_factors: List[str]         # 위험 요소들
    recommended_approach: str       # 추천 접근법
    confidence: float              # 평가 신뢰도 [0-1]

@dataclass
class LLMParsingResult:
    """LLM 파싱 전체 결과"""
    original_command: str
    action_intent: ActionIntent
    target_object: str
    destination: Optional[str]
    physical_properties: ExtractedPhysicalProperties
    control_parameters: ControlParameters
    affordance_assessment: AffordanceAssessment
    processing_time: float
    timestamp: str

# ============================================================================
# LLM 인터페이스 추상화
# ============================================================================

class LLMInterface(ABC):
    """LLM 인터페이스 추상 클래스"""

    @abstractmethod
    def extract_physical_properties(self, command: str) -> ExtractedPhysicalProperties:
        """자연어에서 물리 속성 추출"""
        pass

    @abstractmethod
    def generate_control_parameters(self,
                                  properties: ExtractedPhysicalProperties,
                                  action: ActionIntent) -> ControlParameters:
        """물리 속성 기반 제어 파라미터 생성"""
        pass

    @abstractmethod
    def assess_affordances(self,
                          object_desc: str,
                          properties: ExtractedPhysicalProperties,
                          action: ActionIntent) -> AffordanceAssessment:
        """어포던스 평가"""
        pass

class MockLLMInterface(LLMInterface):
    """목업 LLM 인터페이스 (개발/테스트용)"""

    def __init__(self):
        # 물리 속성 키워드 매핑
        self.mass_keywords = {
            'heavy': ['heavy', 'weighty', 'massive', 'dense'],
            'light': ['light', 'lightweight', 'feather', 'airy'],
            'medium': ['normal', 'regular', 'standard']
        }

        self.friction_keywords = {
            'low': ['slippery', 'smooth', 'slick', 'oily'],
            'high': ['rough', 'grippy', 'textured', 'coarse'],
            'normal': ['normal', 'regular']
        }

        self.stiffness_keywords = {
            'soft': ['soft', 'spongy', 'flexible', 'malleable'],
            'hard': ['hard', 'rigid', 'stiff', 'solid'],
            'medium': ['normal', 'regular']
        }

        self.material_keywords = {
            'metal': ['metal', 'steel', 'iron', 'aluminum'],
            'plastic': ['plastic', 'polymer'],
            'wood': ['wood', 'wooden'],
            'glass': ['glass', 'crystal'],
            'fabric': ['fabric', 'cloth', 'textile']
        }

    def extract_physical_properties(self, command: str) -> ExtractedPhysicalProperties:
        """키워드 기반 물리 속성 추출 (목업)"""
        command_lower = command.lower()

        # 질량 추출
        mass = "medium"
        for mass_type, keywords in self.mass_keywords.items():
            if any(keyword in command_lower for keyword in keywords):
                mass = mass_type
                break

        # 마찰 추출
        friction = "normal"
        for friction_type, keywords in self.friction_keywords.items():
            if any(keyword in command_lower for keyword in keywords):
                friction = friction_type
                break

        # 강성 추출
        stiffness = "medium"
        for stiffness_type, keywords in self.stiffness_keywords.items():
            if any(keyword in command_lower for keyword in keywords):
                stiffness = stiffness_type
                break

        # 재질 추출
        material = "unknown"
        for material_type, keywords in self.material_keywords.items():
            if any(keyword in command_lower for keyword in keywords):
                material = material_type
                break

        # 깨지기 쉬움 판단
        fragility = "normal"
        if any(word in command_lower for word in ['fragile', 'delicate', 'breakable']):
            fragility = "fragile"
        elif any(word in command_lower for word in ['sturdy', 'robust', 'strong']):
            fragility = "robust"

        return ExtractedPhysicalProperties(
            mass=mass,
            friction=friction,
            stiffness=stiffness,
            fragility=fragility,
            material=material,
            confidence=0.85  # 목업 신뢰도
        )

    def generate_control_parameters(self,
                                  properties: ExtractedPhysicalProperties,
                                  action: ActionIntent) -> ControlParameters:
        """물리 속성 기반 제어 파라미터 생성"""

        # 기본값 설정
        grip_force = 0.5
        lift_speed = 0.5
        approach_angle = 0.0
        contact_force = 0.3
        safety_margin = 0.8

        # 질량에 따른 조정
        if properties.mass == "heavy":
            grip_force += 0.3
            lift_speed -= 0.2
            safety_margin += 0.1
        elif properties.mass == "light":
            grip_force -= 0.2
            lift_speed += 0.2

        # 마찰에 따른 조정
        if properties.friction == "low":  # 미끄러움
            grip_force += 0.2
            approach_angle = 15.0  # 더 신중한 접근
            safety_margin += 0.1
        elif properties.friction == "high":  # 거침
            contact_force += 0.2

        # 강성에 따른 조정
        if properties.stiffness == "soft":
            contact_force -= 0.1
            grip_force -= 0.1
        elif properties.stiffness == "hard":
            contact_force += 0.1

        # 깨지기 쉬움에 따른 조정
        if properties.fragility == "fragile":
            grip_force -= 0.2
            lift_speed -= 0.3
            contact_force -= 0.2
            safety_margin += 0.1

        # 범위 제한
        grip_force = np.clip(grip_force, 0.1, 1.0)
        lift_speed = np.clip(lift_speed, 0.1, 1.0)
        contact_force = np.clip(contact_force, 0.1, 1.0)
        safety_margin = np.clip(safety_margin, 0.5, 1.0)

        return ControlParameters(
            grip_force=grip_force,
            lift_speed=lift_speed,
            approach_angle=approach_angle,
            contact_force=contact_force,
            safety_margin=safety_margin,
            execution_time=2.0 / lift_speed  # 속도에 반비례
        )

    def assess_affordances(self,
                          object_desc: str,
                          properties: ExtractedPhysicalProperties,
                          action: ActionIntent) -> AffordanceAssessment:
        """어포던스 평가"""

        affordances = []
        risk_factors = []
        success_probability = 0.8  # 기본값

        # 동작별 어포던스 평가
        if action == ActionIntent.PICK:
            affordances.extend(["graspable", "liftable"])

            if properties.mass == "heavy":
                risk_factors.append("high_weight")
                success_probability -= 0.1

            if properties.friction == "low":
                risk_factors.append("slippery_surface")
                success_probability -= 0.15

            if properties.fragility == "fragile":
                risk_factors.append("breakage_risk")
                success_probability -= 0.1

        elif action == ActionIntent.PLACE:
            affordances.extend(["placeable", "positionable"])

            if properties.stiffness == "soft":
                risk_factors.append("deformation_risk")
                success_probability -= 0.05

        # 추천 접근법
        if len(risk_factors) == 0:
            recommended_approach = "standard_approach"
        elif "slippery_surface" in risk_factors:
            recommended_approach = "increased_grip_careful_approach"
        elif "high_weight" in risk_factors:
            recommended_approach = "slow_controlled_lift"
        elif "breakage_risk" in risk_factors:
            recommended_approach = "gentle_precise_handling"
        else:
            recommended_approach = "cautious_approach"

        return AffordanceAssessment(
            affordances=affordances,
            success_probability=max(0.3, success_probability),  # 최소 30%
            risk_factors=risk_factors,
            recommended_approach=recommended_approach,
            confidence=0.8
        )

# ============================================================================
# LLM-First 파싱 엔진
# ============================================================================

class LLMFirstParser:
    """LLM-First 자연어 파싱 엔진"""

    def __init__(self, llm_interface: LLMInterface, config: Dict[str, Any] = None):
        self.llm = llm_interface
        self.config = config or {}
        self.physics_mapper = UniversalPhysicsMapper(config)

        # 동작 인식 키워드
        self.action_keywords = {
            ActionIntent.PICK: ['pick', 'grab', 'take', 'lift', '들어', '집어', '잡아'],
            ActionIntent.PLACE: ['place', 'put', 'set', 'position', '놓아', '올려', '두어'],
            ActionIntent.MOVE: ['move', 'transport', 'carry', '이동', '옮겨'],
            ActionIntent.PUSH: ['push', 'press', '밀어', '눌러'],
            ActionIntent.PULL: ['pull', 'drag', '당겨', '끌어'],
            ActionIntent.ROTATE: ['rotate', 'turn', 'spin', '돌려', '회전'],
            ActionIntent.INSPECT: ['inspect', 'check', 'examine', '확인', '검사'],
            ActionIntent.STOP: ['stop', 'halt', 'pause', '멈춰', '정지']
        }

    def parse_command(self, command: str) -> LLMParsingResult:
        """자연어 명령 전체 파싱"""
        start_time = time.time()

        try:
            # 1. 동작 의도 추출
            action_intent = self._extract_action_intent(command)
            logger.info(f"추출된 동작 의도: {action_intent}")

            # 2. 대상 객체 및 목적지 추출
            target_object, destination = self._extract_objects(command)
            logger.info(f"대상 객체: {target_object}, 목적지: {destination}")

            # 3. 물리 속성 추출
            physical_properties = self.llm.extract_physical_properties(command)
            logger.info(f"추출된 물리 속성: {asdict(physical_properties)}")

            # 4. 제어 파라미터 생성
            control_parameters = self.llm.generate_control_parameters(
                physical_properties, action_intent
            )
            logger.info(f"생성된 제어 파라미터: {asdict(control_parameters)}")

            # 5. 어포던스 평가
            affordance_assessment = self.llm.assess_affordances(
                target_object, physical_properties, action_intent
            )
            logger.info(f"어포던스 평가: {asdict(affordance_assessment)}")

            processing_time = time.time() - start_time

            result = LLMParsingResult(
                original_command=command,
                action_intent=action_intent,
                target_object=target_object,
                destination=destination,
                physical_properties=physical_properties,
                control_parameters=control_parameters,
                affordance_assessment=affordance_assessment,
                processing_time=processing_time,
                timestamp=datetime.now().isoformat()
            )

            logger.info(f"파싱 완료 - 처리 시간: {processing_time:.3f}초")
            return result

        except Exception as e:
            logger.error(f"명령 파싱 실패: {e}")
            raise

    def _extract_action_intent(self, command: str) -> ActionIntent:
        """동작 의도 추출"""
        command_lower = command.lower()

        for action, keywords in self.action_keywords.items():
            if any(keyword in command_lower for keyword in keywords):
                return action

        # 기본값
        return ActionIntent.MOVE

    def _extract_objects(self, command: str) -> Tuple[str, Optional[str]]:
        """대상 객체와 목적지 추출 (간단한 키워드 기반)"""
        # 목표: 더 정교한 NLP로 개선 예정

        common_objects = [
            'box', 'cup', 'bottle', 'book', 'phone', 'pen',
            '상자', '컵', '병', '책', '전화', '펜', '블록', '공'
        ]

        common_destinations = [
            'table', 'shelf', 'desk', 'floor', 'container',
            '테이블', '선반', '책상', '바닥', '컨테이너'
        ]

        command_lower = command.lower()

        # 객체 찾기
        target_object = "unknown_object"
        for obj in common_objects:
            if obj in command_lower:
                target_object = obj
                break

        # 목적지 찾기
        destination = None
        for dest in common_destinations:
            if dest in command_lower:
                destination = dest
                break

        return target_object, destination

    def validate_result(self, result: LLMParsingResult) -> Dict[str, Any]:
        """파싱 결과 검증"""
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }

        # 처리 시간 검증 (논문 목표: 200ms)
        if result.processing_time > 0.2:
            validation['warnings'].append(f"Processing time {result.processing_time:.3f}s exceeds 200ms target")

        # 신뢰도 검증
        if result.physical_properties.confidence < 0.7:
            validation['warnings'].append(f"Low confidence in physical properties: {result.physical_properties.confidence:.2f}")

        # 어포던스 평가 검증
        if result.affordance_assessment.success_probability < 0.5:
            validation['warnings'].append(f"Low success probability: {result.affordance_assessment.success_probability:.2f}")

        # 제어 파라미터 범위 검증
        params = result.control_parameters
        if not (0.0 <= params.grip_force <= 1.0):
            validation['errors'].append(f"Invalid grip_force: {params.grip_force}")
            validation['is_valid'] = False

        if not (0.0 <= params.lift_speed <= 1.0):
            validation['errors'].append(f"Invalid lift_speed: {params.lift_speed}")
            validation['is_valid'] = False

        return validation

# ============================================================================
# 유틸리티 함수
# ============================================================================

def create_llm_first_parser(config: Dict[str, Any] = None) -> LLMFirstParser:
    """LLM-First 파서 팩토리 함수"""
    # 현재는 목업 사용, 나중에 실제 LLM으로 교체
    mock_llm = MockLLMInterface()
    return LLMFirstParser(mock_llm, config)

def format_parsing_result(result: LLMParsingResult) -> str:
    """파싱 결과를 읽기 쉬운 형태로 포맷"""
    output = []
    output.append(f"원본 명령: {result.original_command}")
    output.append(f"동작 의도: {result.action_intent.value}")
    output.append(f"대상 객체: {result.target_object}")
    if result.destination:
        output.append(f"목적지: {result.destination}")

    output.append("\n물리 속성:")
    props = result.physical_properties
    output.append(f"  질량: {props.mass}")
    output.append(f"  마찰: {props.friction}")
    output.append(f"  강성: {props.stiffness}")
    output.append(f"  재질: {props.material}")
    output.append(f"  신뢰도: {props.confidence:.2f}")

    output.append("\n제어 파라미터:")
    ctrl = result.control_parameters
    output.append(f"  그립력: {ctrl.grip_force:.2f}")
    output.append(f"  들기 속도: {ctrl.lift_speed:.2f}")
    output.append(f"  접근 각도: {ctrl.approach_angle:.1f}°")
    output.append(f"  접촉력: {ctrl.contact_force:.2f}")
    output.append(f"  안전 여유도: {ctrl.safety_margin:.2f}")

    output.append("\n어포던스 평가:")
    afford = result.affordance_assessment
    output.append(f"  성공 확률: {afford.success_probability:.2f}")
    output.append(f"  어포던스: {', '.join(afford.affordances)}")
    if afford.risk_factors:
        output.append(f"  위험 요소: {', '.join(afford.risk_factors)}")
    output.append(f"  추천 접근법: {afford.recommended_approach}")

    output.append(f"\n처리 시간: {result.processing_time:.3f}초")

    return "\n".join(output)

if __name__ == "__main__":
    # 테스트 실행
    parser = create_llm_first_parser({"debug": True})

    test_commands = [
        "무거운 금속 상자를 선반에 올려놔",
        "미끄러운 컵을 조심스럽게 잡아",
        "말랑한 스펀지를 부드럽게 눌러",
        "깨지기 쉬운 유리병을 테이블로 옮겨"
    ]

    for command in test_commands:
        print(f"\n{'='*60}")
        print(f"테스트 명령: {command}")
        print(f"{'='*60}")

        result = parser.parse_command(command)
        print(format_parsing_result(result))

        validation = parser.validate_result(result)
        print(f"\n검증 결과: {'✓ 통과' if validation['is_valid'] else '✗ 실패'}")
        if validation['warnings']:
            print("경고:", ", ".join(validation['warnings']))
        if validation['errors']:
            print("오류:", ", ".join(validation['errors']))