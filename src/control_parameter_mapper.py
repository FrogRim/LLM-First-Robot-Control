#!/usr/bin/env python3
"""
제어 파라미터 매핑 시스템 (Control Parameter Mapping System)

졸업논문: "LLM-First 기반 물리 속성 추출 로봇 제어"
물리 속성을 로봇 제어 파라미터로 변환하는 매핑 로직
- grip_force, lift_speed, approach_angle 등을 ROS2 메시지 형태로 전달
- 응답시간 200ms 이내 목표
"""

import time
import json
import math
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
from queue import Queue, Empty
from collections import deque

# 기존 모듈 연동
from llm_first_layer import ExtractedPhysicalProperties, ControlParameters, ActionIntent, AffordanceAssessment
from data_abstraction_layer import RobotConfiguration, TrajectoryData

logger = logging.getLogger(__name__)

# ============================================================================
# 제어 파라미터 데이터 구조
# ============================================================================

class ControlMode(Enum):
    """제어 모드"""
    POSITION = "position"           # 위치 제어
    VELOCITY = "velocity"           # 속도 제어
    FORCE = "force"                 # 힘 제어
    IMPEDANCE = "impedance"         # 임피던스 제어
    HYBRID = "hybrid"               # 하이브리드 제어

class SafetyLevel(Enum):
    """안전 수준"""
    MINIMAL = "minimal"             # 최소 안전
    STANDARD = "standard"           # 표준 안전
    HIGH = "high"                   # 높은 안전
    MAXIMUM = "maximum"             # 최대 안전

@dataclass
class RobotControlCommand:
    """로봇 제어 명령"""
    # 그립 제어
    grip_force: float = 0.5         # 그립력 [0-1]
    grip_speed: float = 0.5         # 그립 속도 [0-1]
    grip_position: float = 0.5      # 그립 위치 [0-1]

    # 팔 움직임 제어
    joint_positions: List[float] = field(default_factory=lambda: [0.0] * 7)  # 관절 위치 [rad]
    joint_velocities: List[float] = field(default_factory=lambda: [0.0] * 7)  # 관절 속도 [rad/s]
    joint_torques: List[float] = field(default_factory=lambda: [0.0] * 7)    # 관절 토크 [Nm]

    # 끝단 효과기 제어
    end_effector_position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # [x, y, z]
    end_effector_orientation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # [roll, pitch, yaw]
    end_effector_force: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # [fx, fy, fz]
    end_effector_torque: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # [tx, ty, tz]

    # 움직임 파라미터
    approach_angle: float = 0.0     # 접근 각도 [도]
    lift_speed: float = 0.5         # 들기 속도 [0-1]
    contact_force: float = 0.3      # 접촉력 [0-1]

    # 안전 및 제어 설정
    safety_margin: float = 0.8      # 안전 여유도 [0-1]
    control_mode: ControlMode = ControlMode.HYBRID
    safety_level: SafetyLevel = SafetyLevel.STANDARD

    # 시간 제약
    execution_timeout: float = 10.0  # 실행 타임아웃 [초]
    trajectory_duration: float = 2.0  # 궤적 지속시간 [초]

    # 메타데이터
    command_id: str = ""
    timestamp: float = 0.0
    priority: int = 0               # 우선순위 (높을수록 우선)

@dataclass
class ControlParameters_Extended:
    """확장된 제어 파라미터 (논문 요구사항)"""
    # 기본 파라미터 (논문에서 요구)
    grip_force: float = 0.5         # 그립력 [0-1]
    lift_speed: float = 0.5         # 들기 속도 [0-1]
    approach_angle: float = 0.0     # 접근 각도 [도]

    # 추가 세부 파라미터
    contact_force: float = 0.3      # 접촉력 [0-1]
    safety_margin: float = 0.8      # 안전 여유도 [0-1]

    # 동적 파라미터
    acceleration_limit: float = 1.0  # 가속도 제한 [m/s²]
    jerk_limit: float = 2.0         # 저크 제한 [m/s³]
    force_limit: float = 50.0       # 힘 제한 [N]

    # 어댑티브 파라미터
    stiffness_ratio: float = 0.5    # 강성 비율 [0-1]
    damping_ratio: float = 0.7      # 감쇠 비율 [0-1]

    # 센서 기반 파라미터
    force_threshold: float = 5.0    # 힘 임계값 [N]
    position_tolerance: float = 0.001  # 위치 허용오차 [m]

    # 타이밍 파라미터
    pre_grasp_delay: float = 0.1    # 그립 전 지연 [초]
    post_grasp_delay: float = 0.2   # 그립 후 지연 [초]

# ============================================================================
# 매핑 룰 엔진
# ============================================================================

class MappingRule:
    """매핑 규칙"""

    def __init__(self,
                 name: str,
                 condition: callable,
                 mapping_func: callable,
                 priority: int = 0):
        self.name = name
        self.condition = condition  # 적용 조건 함수
        self.mapping_func = mapping_func  # 매핑 함수
        self.priority = priority

class ControlParameterMappingEngine:
    """제어 파라미터 매핑 엔진"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.mapping_rules = []
        self.performance_cache = {}

        # 매핑 룰 초기화
        self._initialize_mapping_rules()

        # 성능 최적화를 위한 설정
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.max_cache_size = self.config.get('max_cache_size', 1000)

    def _initialize_mapping_rules(self):
        """매핑 룰 초기화"""

        # 1. 질량 기반 규칙
        self.mapping_rules.append(MappingRule(
            name="heavy_object_rule",
            condition=lambda props, action: props.mass == "heavy",
            mapping_func=self._map_heavy_object,
            priority=10
        ))

        self.mapping_rules.append(MappingRule(
            name="light_object_rule",
            condition=lambda props, action: props.mass == "light",
            mapping_func=self._map_light_object,
            priority=10
        ))

        # 2. 마찰 기반 규칙
        self.mapping_rules.append(MappingRule(
            name="slippery_surface_rule",
            condition=lambda props, action: props.friction == "low",
            mapping_func=self._map_slippery_surface,
            priority=15
        ))

        self.mapping_rules.append(MappingRule(
            name="rough_surface_rule",
            condition=lambda props, action: props.friction == "high",
            mapping_func=self._map_rough_surface,
            priority=8
        ))

        # 3. 깨지기 쉬움 기반 규칙
        self.mapping_rules.append(MappingRule(
            name="fragile_object_rule",
            condition=lambda props, action: props.fragility == "fragile",
            mapping_func=self._map_fragile_object,
            priority=20  # 높은 우선순위
        ))

        # 4. 재료 기반 규칙
        self.mapping_rules.append(MappingRule(
            name="metal_object_rule",
            condition=lambda props, action: props.material == "metal",
            mapping_func=self._map_metal_object,
            priority=5
        ))

        self.mapping_rules.append(MappingRule(
            name="glass_object_rule",
            condition=lambda props, action: props.material == "glass",
            mapping_func=self._map_glass_object,
            priority=18
        ))

        # 5. 동작 기반 규칙
        self.mapping_rules.append(MappingRule(
            name="pick_action_rule",
            condition=lambda props, action: action == ActionIntent.PICK,
            mapping_func=self._map_pick_action,
            priority=12
        ))

        self.mapping_rules.append(MappingRule(
            name="place_action_rule",
            condition=lambda props, action: action == ActionIntent.PLACE,
            mapping_func=self._map_place_action,
            priority=12
        ))

        # 우선순위로 정렬
        self.mapping_rules.sort(key=lambda rule: rule.priority, reverse=True)

    def map_to_control_parameters(self,
                                properties: ExtractedPhysicalProperties,
                                action: ActionIntent,
                                affordance: AffordanceAssessment = None,
                                context: Dict[str, Any] = None) -> ControlParameters_Extended:
        """물리 속성을 제어 파라미터로 매핑"""

        start_time = time.time()

        # 캐시 확인
        cache_key = self._generate_cache_key(properties, action, context)
        if self.cache_enabled and cache_key in self.performance_cache:
            logger.debug("캐시된 매핑 결과 사용")
            return self.performance_cache[cache_key]

        # 기본 파라미터로 시작
        control_params = ControlParameters_Extended()

        # 적용 가능한 규칙들 찾기
        applicable_rules = []
        for rule in self.mapping_rules:
            try:
                if rule.condition(properties, action):
                    applicable_rules.append(rule)
            except Exception as e:
                logger.warning(f"규칙 {rule.name} 조건 검사 실패: {e}")

        logger.info(f"적용 가능한 규칙 수: {len(applicable_rules)}")

        # 규칙들을 순서대로 적용
        for rule in applicable_rules:
            try:
                control_params = rule.mapping_func(control_params, properties, action, affordance, context)
                logger.debug(f"규칙 {rule.name} 적용 완료")
            except Exception as e:
                logger.error(f"규칙 {rule.name} 적용 실패: {e}")

        # 안전성 검증 및 조정
        control_params = self._apply_safety_constraints(control_params, properties, affordance)

        # 성능 최적화 (200ms 목표)
        control_params = self._optimize_for_performance(control_params, properties)

        processing_time = time.time() - start_time
        logger.info(f"매핑 완료 - 처리 시간: {processing_time:.3f}초")

        # 캐시 저장
        if self.cache_enabled:
            self._update_cache(cache_key, control_params)

        return control_params

    def _map_heavy_object(self, params: ControlParameters_Extended, props: ExtractedPhysicalProperties,
                         action: ActionIntent, affordance: AffordanceAssessment, context: Dict[str, Any]) -> ControlParameters_Extended:
        """무거운 객체 매핑"""
        params.grip_force = min(1.0, params.grip_force + 0.3)
        params.lift_speed = max(0.1, params.lift_speed - 0.2)
        params.safety_margin = min(1.0, params.safety_margin + 0.1)
        params.force_limit = min(100.0, params.force_limit + 20.0)
        params.acceleration_limit = max(0.3, params.acceleration_limit - 0.3)
        return params

    def _map_light_object(self, params: ControlParameters_Extended, props: ExtractedPhysicalProperties,
                         action: ActionIntent, affordance: AffordanceAssessment, context: Dict[str, Any]) -> ControlParameters_Extended:
        """가벼운 객체 매핑"""
        params.grip_force = max(0.1, params.grip_force - 0.2)
        params.lift_speed = min(1.0, params.lift_speed + 0.2)
        params.acceleration_limit = min(2.0, params.acceleration_limit + 0.4)
        params.force_threshold = max(1.0, params.force_threshold - 2.0)
        return params

    def _map_slippery_surface(self, params: ControlParameters_Extended, props: ExtractedPhysicalProperties,
                             action: ActionIntent, affordance: AffordanceAssessment, context: Dict[str, Any]) -> ControlParameters_Extended:
        """미끄러운 표면 매핑"""
        params.grip_force = min(1.0, params.grip_force + 0.25)
        params.approach_angle = 15.0  # 더 신중한 접근
        params.contact_force = min(1.0, params.contact_force + 0.2)
        params.safety_margin = min(1.0, params.safety_margin + 0.15)
        params.pre_grasp_delay = 0.2  # 접촉 전 충분한 준비
        return params

    def _map_rough_surface(self, params: ControlParameters_Extended, props: ExtractedPhysicalProperties,
                          action: ActionIntent, affordance: AffordanceAssessment, context: Dict[str, Any]) -> ControlParameters_Extended:
        """거친 표면 매핑"""
        params.contact_force = min(1.0, params.contact_force + 0.1)
        params.approach_angle = max(-10.0, params.approach_angle - 5.0)
        return params

    def _map_fragile_object(self, params: ControlParameters_Extended, props: ExtractedPhysicalProperties,
                           action: ActionIntent, affordance: AffordanceAssessment, context: Dict[str, Any]) -> ControlParameters_Extended:
        """깨지기 쉬운 객체 매핑"""
        params.grip_force = max(0.1, params.grip_force - 0.25)
        params.lift_speed = max(0.1, params.lift_speed - 0.3)
        params.contact_force = max(0.1, params.contact_force - 0.2)
        params.safety_margin = min(1.0, params.safety_margin + 0.2)
        params.force_limit = max(10.0, params.force_limit - 30.0)
        params.acceleration_limit = max(0.2, params.acceleration_limit - 0.5)
        params.jerk_limit = max(0.5, params.jerk_limit - 1.0)
        params.force_threshold = max(1.0, params.force_threshold - 3.0)
        params.position_tolerance = min(0.005, params.position_tolerance + 0.002)
        params.pre_grasp_delay = 0.3
        params.post_grasp_delay = 0.4
        return params

    def _map_metal_object(self, params: ControlParameters_Extended, props: ExtractedPhysicalProperties,
                         action: ActionIntent, affordance: AffordanceAssessment, context: Dict[str, Any]) -> ControlParameters_Extended:
        """금속 객체 매핑"""
        params.stiffness_ratio = min(1.0, params.stiffness_ratio + 0.2)
        params.force_limit = min(100.0, params.force_limit + 10.0)
        return params

    def _map_glass_object(self, params: ControlParameters_Extended, props: ExtractedPhysicalProperties,
                         action: ActionIntent, affordance: AffordanceAssessment, context: Dict[str, Any]) -> ControlParameters_Extended:
        """유리 객체 매핑"""
        # 깨지기 쉬운 객체와 유사하지만 더 세밀한 제어
        params.grip_force = max(0.1, params.grip_force - 0.3)
        params.lift_speed = max(0.05, params.lift_speed - 0.4)
        params.contact_force = max(0.05, params.contact_force - 0.25)
        params.safety_margin = 1.0  # 최대 안전
        params.force_limit = max(5.0, params.force_limit - 40.0)
        params.acceleration_limit = max(0.1, params.acceleration_limit - 0.7)
        params.damping_ratio = min(1.0, params.damping_ratio + 0.2)
        return params

    def _map_pick_action(self, params: ControlParameters_Extended, props: ExtractedPhysicalProperties,
                        action: ActionIntent, affordance: AffordanceAssessment, context: Dict[str, Any]) -> ControlParameters_Extended:
        """집기 동작 매핑"""
        params.approach_angle = abs(params.approach_angle)  # 위에서 접근
        if affordance and affordance.success_probability < 0.6:
            params.pre_grasp_delay = 0.25  # 더 신중하게
        return params

    def _map_place_action(self, params: ControlParameters_Extended, props: ExtractedPhysicalProperties,
                         action: ActionIntent, affordance: AffordanceAssessment, context: Dict[str, Any]) -> ControlParameters_Extended:
        """놓기 동작 매핑"""
        params.lift_speed = max(0.2, params.lift_speed - 0.1)  # 놓을 때는 천천히
        params.contact_force = max(0.1, params.contact_force - 0.1)
        params.post_grasp_delay = 0.3  # 놓은 후 충분한 대기
        return params

    def _apply_safety_constraints(self, params: ControlParameters_Extended,
                                 properties: ExtractedPhysicalProperties,
                                 affordance: AffordanceAssessment = None) -> ControlParameters_Extended:
        """안전성 제약 조건 적용"""

        # 기본 안전 범위 확인
        params.grip_force = np.clip(params.grip_force, 0.05, 1.0)
        params.lift_speed = np.clip(params.lift_speed, 0.05, 1.0)
        params.contact_force = np.clip(params.contact_force, 0.05, 1.0)
        params.safety_margin = np.clip(params.safety_margin, 0.5, 1.0)

        # 어포던스 기반 안전 조정
        if affordance:
            if affordance.success_probability < 0.5:
                params.safety_margin = max(0.9, params.safety_margin)
                params.lift_speed = min(0.3, params.lift_speed)

            if "breakage_risk" in affordance.risk_factors:
                params.force_limit = min(params.force_limit, 15.0)
                params.grip_force = min(params.grip_force, 0.3)

        return params

    def _optimize_for_performance(self, params: ControlParameters_Extended,
                                 properties: ExtractedPhysicalProperties) -> ControlParameters_Extended:
        """성능 최적화 (200ms 목표)"""

        # 신뢰도가 높으면 더 적극적인 파라미터 사용
        if properties.confidence > 0.8:
            params.pre_grasp_delay = max(0.05, params.pre_grasp_delay - 0.05)
            params.post_grasp_delay = max(0.1, params.post_grasp_delay - 0.1)

        # 불확실성이 높으면 보수적 접근
        elif properties.confidence < 0.6:
            params.pre_grasp_delay = min(0.4, params.pre_grasp_delay + 0.1)
            params.safety_margin = min(1.0, params.safety_margin + 0.1)

        return params

    def _generate_cache_key(self, properties: ExtractedPhysicalProperties,
                           action: ActionIntent, context: Dict[str, Any]) -> str:
        """캐시 키 생성"""
        key_data = {
            'mass': properties.mass,
            'friction': properties.friction,
            'stiffness': properties.stiffness,
            'fragility': properties.fragility,
            'material': properties.material,
            'action': action.value,
            'context': str(context) if context else None
        }
        return str(hash(str(key_data)))

    def _update_cache(self, key: str, params: ControlParameters_Extended):
        """캐시 업데이트"""
        if len(self.performance_cache) >= self.max_cache_size:
            # LRU 방식으로 오래된 항목 제거
            oldest_key = next(iter(self.performance_cache))
            del self.performance_cache[oldest_key]

        self.performance_cache[key] = params

# ============================================================================
# ROS2 메시지 생성기
# ============================================================================

class ROS2MessageGenerator:
    """ROS2 메시지 생성기"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def generate_control_message(self, params: ControlParameters_Extended,
                                robot_config: RobotConfiguration = None) -> Dict[str, Any]:
        """제어 메시지 생성 (ROS2 호환)"""

        # geometry_msgs/Twist 형태의 메시지 구조
        twist_msg = {
            "linear": {
                "x": 0.0,
                "y": 0.0,
                "z": params.lift_speed * 0.1  # 0.1 m/s 최대 속도
            },
            "angular": {
                "x": 0.0,
                "y": 0.0,
                "z": math.radians(params.approach_angle)
            }
        }

        # sensor_msgs/JointState 형태의 그립 메시지
        gripper_msg = {
            "header": {
                "stamp": {"sec": int(time.time()), "nanosec": 0},
                "frame_id": "gripper_frame"
            },
            "name": ["gripper_joint"],
            "position": [params.grip_force],
            "velocity": [0.1],  # 고정 속도
            "effort": [params.contact_force * 10.0]  # N 단위
        }

        # 사용자 정의 제어 메시지
        control_msg = {
            "header": {
                "stamp": {"sec": int(time.time()), "nanosec": 0},
                "frame_id": "robot_base"
            },
            "control_parameters": {
                "grip_force": params.grip_force,
                "lift_speed": params.lift_speed,
                "approach_angle": params.approach_angle,
                "contact_force": params.contact_force,
                "safety_margin": params.safety_margin
            },
            "safety_limits": {
                "force_limit": params.force_limit,
                "acceleration_limit": params.acceleration_limit,
                "jerk_limit": params.jerk_limit
            },
            "timing_parameters": {
                "pre_grasp_delay": params.pre_grasp_delay,
                "post_grasp_delay": params.post_grasp_delay,
                "execution_timeout": 10.0
            }
        }

        return {
            "twist": twist_msg,
            "gripper": gripper_msg,
            "control": control_msg
        }

    def generate_trajectory_message(self, params: ControlParameters_Extended,
                                  start_pose: List[float], target_pose: List[float],
                                  robot_config: RobotConfiguration = None) -> Dict[str, Any]:
        """궤적 메시지 생성"""

        # trajectory_msgs/JointTrajectory 형태
        trajectory_msg = {
            "header": {
                "stamp": {"sec": int(time.time()), "nanosec": 0},
                "frame_id": "robot_base"
            },
            "joint_names": robot_config.joint_names if robot_config else [f"joint_{i}" for i in range(7)],
            "points": []
        }

        # 간단한 선형 궤적 생성 (실제로는 더 정교한 계획 필요)
        num_points = max(10, int(params.lift_speed * 50))  # 속도에 따른 포인트 수
        duration = 2.0 / params.lift_speed  # 속도에 반비례하는 지속시간

        for i in range(num_points + 1):
            t = i / num_points

            # 선형 보간
            if robot_config:
                positions = [
                    start_pose[j] + t * (target_pose[j] - start_pose[j])
                    for j in range(len(start_pose))
                ]
            else:
                positions = [0.0] * 7  # 기본 7-DOF

            point = {
                "positions": positions,
                "velocities": [0.0] * len(positions),
                "accelerations": [0.0] * len(positions),
                "effort": [0.0] * len(positions),
                "time_from_start": {
                    "sec": int(t * duration),
                    "nanosec": int((t * duration % 1) * 1e9)
                }
            }

            trajectory_msg["points"].append(point)

        return trajectory_msg

# ============================================================================
# 성능 검증기
# ============================================================================

class PerformanceValidator:
    """성능 검증기"""

    def __init__(self, target_response_time: float = 0.2):
        self.target_response_time = target_response_time
        self.response_times = deque(maxlen=100)  # 최근 100개 기록

    def validate_response_time(self, processing_time: float) -> Dict[str, Any]:
        """응답 시간 검증"""

        self.response_times.append(processing_time)

        result = {
            'current_time': processing_time,
            'target_time': self.target_response_time,
            'meets_target': processing_time <= self.target_response_time,
            'average_time': np.mean(self.response_times) if self.response_times else 0.0,
            'max_time': max(self.response_times) if self.response_times else 0.0,
            'success_rate': sum(1 for t in self.response_times if t <= self.target_response_time) / len(self.response_times) if self.response_times else 0.0
        }

        return result

    def validate_control_parameters(self, params: ControlParameters_Extended) -> Dict[str, Any]:
        """제어 파라미터 검증"""

        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }

        # 범위 검증
        if not (0.0 <= params.grip_force <= 1.0):
            validation['errors'].append(f"grip_force out of range: {params.grip_force}")
            validation['is_valid'] = False

        if not (0.0 <= params.lift_speed <= 1.0):
            validation['errors'].append(f"lift_speed out of range: {params.lift_speed}")
            validation['is_valid'] = False

        # 안전성 검증
        if params.force_limit > 100.0:
            validation['warnings'].append(f"High force limit: {params.force_limit}N")

        if params.safety_margin < 0.5:
            validation['warnings'].append(f"Low safety margin: {params.safety_margin}")

        return validation

# ============================================================================
# 통합 인터페이스
# ============================================================================

def create_control_mapper(config: Dict[str, Any] = None) -> ControlParameterMappingEngine:
    """제어 파라미터 매퍼 팩토리"""
    return ControlParameterMappingEngine(config)

def create_ros2_generator(config: Dict[str, Any] = None) -> ROS2MessageGenerator:
    """ROS2 메시지 생성기 팩토리"""
    return ROS2MessageGenerator(config)

if __name__ == "__main__":
    # 테스트 실행
    mapper = create_control_mapper({"debug": True, "cache_enabled": True})
    ros2_gen = ROS2MessageGenerator()
    validator = PerformanceValidator()

    from llm_first_layer import ExtractedPhysicalProperties, ActionIntent, AffordanceAssessment

    test_cases = [
        {
            'properties': ExtractedPhysicalProperties(
                mass="heavy", friction="low", stiffness="hard",
                fragility="normal", material="metal", confidence=0.85
            ),
            'action': ActionIntent.PICK,
            'affordance': AffordanceAssessment(
                affordances=["graspable", "liftable"],
                success_probability=0.75,
                risk_factors=["slippery_surface"],
                recommended_approach="careful_approach",
                confidence=0.8
            ),
            'description': "무거운 미끄러운 금속 상자"
        },
        {
            'properties': ExtractedPhysicalProperties(
                mass="light", friction="normal", stiffness="hard",
                fragility="fragile", material="glass", confidence=0.9
            ),
            'action': ActionIntent.PICK,
            'affordance': AffordanceAssessment(
                affordances=["graspable", "breakable"],
                success_probability=0.6,
                risk_factors=["breakage_risk"],
                recommended_approach="gentle_approach",
                confidence=0.85
            ),
            'description': "깨지기 쉬운 유리컵"
        }
    ]

    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"테스트 케이스 {i+1}: {test_case['description']}")
        print(f"{'='*60}")

        start_time = time.time()

        # 매핑 실행
        control_params = mapper.map_to_control_parameters(
            test_case['properties'],
            test_case['action'],
            test_case['affordance']
        )

        processing_time = time.time() - start_time

        # 성능 검증
        time_validation = validator.validate_response_time(processing_time)
        param_validation = validator.validate_control_parameters(control_params)

        print(f"제어 파라미터:")
        print(f"  그립력: {control_params.grip_force:.3f}")
        print(f"  들기 속도: {control_params.lift_speed:.3f}")
        print(f"  접근 각도: {control_params.approach_angle:.1f}°")
        print(f"  접촉력: {control_params.contact_force:.3f}")
        print(f"  안전 여유도: {control_params.safety_margin:.3f}")
        print(f"  힘 제한: {control_params.force_limit:.1f}N")

        print(f"\n성능 검증:")
        print(f"  처리 시간: {processing_time:.3f}초")
        print(f"  목표 달성: {'✓' if time_validation['meets_target'] else '✗'}")
        print(f"  파라미터 유효성: {'✓' if param_validation['is_valid'] else '✗'}")

        if param_validation['warnings']:
            print(f"  경고: {', '.join(param_validation['warnings'])}")

        # ROS2 메시지 생성 테스트
        ros2_msg = ros2_gen.generate_control_message(control_params)
        print(f"\nROS2 메시지 생성: ✓")
        print(f"  그립 위치: {ros2_msg['gripper']['position'][0]:.3f}")
        print(f"  리프트 속도: {ros2_msg['twist']['linear']['z']:.3f} m/s")