"""
물리 매핑 계층 (Physics Mapping Layer)

Isaac Sim ↔ Genesis AI 간의 물리 매개변수 변환 로직
물리 엔진별 특성을 고려한 정확한 변환 수행
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging
from pathlib import Path

from data_abstraction_layer import (
    StandardDataSchema, PhysicalProperties, RobotConfiguration,
    TrajectoryData, PhysicsEngine, RobotType
)


# ============================================================================
# 물리 엔진별 매개변수 매핑 상수
# ============================================================================

class PhysicsConstants:
    """물리 상수 및 변환 팩터"""

    # 단위 변환
    METER_TO_METER = 1.0
    KG_TO_KG = 1.0
    RADIAN_TO_RADIAN = 1.0

    # 물리 엔진별 기본값
    ISAAC_SIM_DEFAULTS = {
        'gravity': np.array([0, 0, -9.81]),
        'time_step': 0.0083,  # 120 Hz
        'solver_iterations': 4,
        'friction_model': 'coulomb',
        'contact_stiffness': 1e6,
        'contact_damping': 1e3
    }

    GENESIS_AI_DEFAULTS = {
        'gravity': np.array([0, 0, -9.81]),
        'time_step': 0.01,   # 100 Hz
        'solver_iterations': 10,
        'friction_model': 'box',
        'contact_stiffness': 1e5,
        'contact_damping': 1e2
    }

    # 매개변수 변환 매트릭스
    ISAAC_TO_GENESIS_SCALING = {
        'mass': 1.0,
        'friction': 0.8,  # Genesis AI에서 더 높은 마찰 표현
        'restitution': 1.0,
        'stiffness': 0.1,  # Genesis AI는 더 유연한 접촉 모델
        'damping': 0.1,
        'joint_stiffness': 0.5,
        'joint_damping': 2.0
    }


@dataclass
class PhysicsEngineProfile:
    """물리 엔진 프로필"""
    engine_type: PhysicsEngine
    version: str
    supported_features: List[str]
    precision_level: str  # 'low', 'medium', 'high'
    simulation_frequency: float  # Hz

    # 수치적 특성
    numerical_stability: float  # 0.0 ~ 1.0
    convergence_threshold: float
    maximum_iterations: int

    # 접촉 모델
    contact_model: str
    friction_model: str
    collision_detection: str


# ============================================================================
# 물리 매개변수 변환 인터페이스
# ============================================================================

class PhysicsMapper(ABC):
    """물리 매개변수 변환 추상 인터페이스"""

    @abstractmethod
    def map_properties(self, source_props: PhysicalProperties,
                       source_engine: PhysicsEngine,
                       target_engine: PhysicsEngine) -> PhysicalProperties:
        """물리 속성 변환"""
        pass

    @abstractmethod
    def map_robot_dynamics(self, robot_config: RobotConfiguration,
                           source_engine: PhysicsEngine,
                           target_engine: PhysicsEngine) -> RobotConfiguration:
        """로봇 동역학 매개변수 변환"""
        pass

    @abstractmethod
    def validate_conversion(self, original: PhysicalProperties,
                           converted: PhysicalProperties) -> Dict[str, Any]:
        """변환 결과 검증"""
        pass

    @abstractmethod
    def estimate_accuracy(self, source_engine: PhysicsEngine,
                         target_engine: PhysicsEngine) -> float:
        """변환 정확도 추정"""
        pass


class UniversalPhysicsMapper(PhysicsMapper):
    """범용 물리 매개변수 변환기"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 물리 엔진 프로필 로딩
        self.engine_profiles = self._load_engine_profiles()

        # 변환 매트릭스 초기화
        self.conversion_matrices = self._initialize_conversion_matrices()

        # 검증 임계값
        self.validation_thresholds = {
            'mass_tolerance': 0.01,        # 1%
            'friction_tolerance': 0.1,     # 10%
            'stiffness_tolerance': 0.2,    # 20%
            'energy_conservation': 0.05    # 5%
        }

    def map_properties(self, source_props: PhysicalProperties,
                       source_engine: PhysicsEngine,
                       target_engine: PhysicsEngine) -> PhysicalProperties:
        """물리 속성 변환"""

        if source_engine == target_engine:
            return source_props

        self.logger.info(f"Converting physics properties: {source_engine} → {target_engine}")

        # 변환 매트릭스 선택
        conversion_key = f"{source_engine.value}_to_{target_engine.value}"
        conversion_factors = self.conversion_matrices.get(conversion_key, {})

        # 기본 물리 속성 변환
        converted_props = PhysicalProperties(
            mass=source_props.mass * conversion_factors.get('mass', 1.0),
            friction_coefficient=self._convert_friction(
                source_props.friction_coefficient, source_engine, target_engine
            ),
            restitution=self._convert_restitution(
                source_props.restitution, source_engine, target_engine
            ),
            linear_damping=source_props.linear_damping * conversion_factors.get('linear_damping', 1.0),
            angular_damping=source_props.angular_damping * conversion_factors.get('angular_damping', 1.0),

            # 고급 속성 변환
            inertia_tensor=self._convert_inertia_tensor(source_props.inertia_tensor),
            center_of_mass=source_props.center_of_mass,  # 좌표계 동일 가정
            contact_stiffness=self._convert_contact_stiffness(
                source_props.contact_stiffness, source_engine, target_engine
            ),
            contact_damping=self._convert_contact_damping(
                source_props.contact_damping, source_engine, target_engine
            ),

            # 메타데이터 보존
            material_type=source_props.material_type,
            surface_roughness=source_props.surface_roughness,
            density=source_props.density
        )

        # 물리적 타당성 검증
        self._validate_physical_properties(converted_props)

        return converted_props

    def map_robot_dynamics(self, robot_config: RobotConfiguration,
                          source_engine: PhysicsEngine,
                          target_engine: PhysicsEngine) -> RobotConfiguration:
        """로봇 동역학 매개변수 변환"""

        if source_engine == target_engine:
            return robot_config

        self.logger.info(f"Converting robot dynamics: {source_engine} → {target_engine}")

        conversion_key = f"{source_engine.value}_to_{target_engine.value}"
        conversion_factors = self.conversion_matrices.get(conversion_key, {})

        # 동역학 매개변수 변환
        converted_config = RobotConfiguration(
            robot_type=robot_config.robot_type,
            joint_count=robot_config.joint_count,
            joint_names=robot_config.joint_names.copy(),
            joint_types=robot_config.joint_types.copy(),
            joint_limits=robot_config.joint_limits.copy(),

            # 기구학 정보 (보존)
            base_pose=robot_config.base_pose.copy(),
            end_effector_offset=robot_config.end_effector_offset.copy(),
            link_lengths=robot_config.link_lengths.copy(),
            link_masses=robot_config.link_masses.copy(),

            # 동역학 매개변수 변환
            joint_stiffness=[
                stiffness * conversion_factors.get('joint_stiffness', 1.0)
                for stiffness in robot_config.joint_stiffness
            ],
            joint_damping=[
                damping * conversion_factors.get('joint_damping', 1.0)
                for damping in robot_config.joint_damping
            ],
            max_joint_velocities=robot_config.max_joint_velocities.copy(),
            max_joint_torques=[
                torque * conversion_factors.get('max_torque', 1.0)
                for torque in robot_config.max_joint_torques
            ]
        )

        return converted_config

    def _convert_friction(self, friction: float, source: PhysicsEngine,
                         target: PhysicsEngine) -> float:
        """마찰계수 변환"""
        if source == PhysicsEngine.ISAAC_SIM and target == PhysicsEngine.GENESIS_AI:
            # Isaac Sim의 Coulomb 마찰 → Genesis AI의 Box 마찰
            return friction * PhysicsConstants.ISAAC_TO_GENESIS_SCALING['friction']
        elif source == PhysicsEngine.GENESIS_AI and target == PhysicsEngine.ISAAC_SIM:
            # 역변환
            return friction / PhysicsConstants.ISAAC_TO_GENESIS_SCALING['friction']
        else:
            return friction

    def _convert_restitution(self, restitution: float, source: PhysicsEngine,
                           target: PhysicsEngine) -> float:
        """탄성계수 변환"""
        # 대부분의 물리 엔진에서 탄성계수는 표준화되어 있음
        if source == PhysicsEngine.ISAAC_SIM and target == PhysicsEngine.GENESIS_AI:
            # Genesis AI는 더 보수적인 탄성 모델 사용
            return min(restitution * 0.9, 1.0)
        elif source == PhysicsEngine.GENESIS_AI and target == PhysicsEngine.ISAAC_SIM:
            return min(restitution / 0.9, 1.0)
        else:
            return restitution

    def _convert_contact_stiffness(self, stiffness: Optional[float],
                                  source: PhysicsEngine,
                                  target: PhysicsEngine) -> Optional[float]:
        """접촉 강성 변환"""
        if stiffness is None:
            return None

        if source == PhysicsEngine.ISAAC_SIM and target == PhysicsEngine.GENESIS_AI:
            # Genesis AI는 더 유연한 접촉 모델 선호
            return stiffness * PhysicsConstants.ISAAC_TO_GENESIS_SCALING['stiffness']
        elif source == PhysicsEngine.GENESIS_AI and target == PhysicsEngine.ISAAC_SIM:
            return stiffness / PhysicsConstants.ISAAC_TO_GENESIS_SCALING['stiffness']
        else:
            return stiffness

    def _convert_contact_damping(self, damping: Optional[float],
                               source: PhysicsEngine,
                               target: PhysicsEngine) -> Optional[float]:
        """접촉 감쇠 변환"""
        if damping is None:
            return None

        if source == PhysicsEngine.ISAAC_SIM and target == PhysicsEngine.GENESIS_AI:
            return damping * PhysicsConstants.ISAAC_TO_GENESIS_SCALING['damping']
        elif source == PhysicsEngine.GENESIS_AI and target == PhysicsEngine.ISAAC_SIM:
            return damping / PhysicsConstants.ISAAC_TO_GENESIS_SCALING['damping']
        else:
            return damping

    def _convert_inertia_tensor(self, inertia: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """관성 텐서 변환"""
        if inertia is None:
            return None

        # 관성 텐서는 질량 분포를 나타내므로 좌표계가 동일하면 보존
        return inertia.copy()

    def _validate_physical_properties(self, props: PhysicalProperties):
        """물리 속성 유효성 검증"""
        if props.mass <= 0:
            raise ValueError(f"Invalid mass: {props.mass}")

        if not (0 <= props.friction_coefficient <= 2.0):
            self.logger.warning(f"Unusual friction coefficient: {props.friction_coefficient}")

        if not (0 <= props.restitution <= 1.0):
            raise ValueError(f"Invalid restitution: {props.restitution}")

        if props.linear_damping < 0 or props.angular_damping < 0:
            raise ValueError("Damping values must be non-negative")

    def validate_conversion(self, original: PhysicalProperties,
                          converted: PhysicalProperties) -> Dict[str, Any]:
        """변환 결과 검증"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'accuracy_score': 0.0,
            'conservation_metrics': {}
        }

        # 질량 보존 검사
        mass_error = abs(converted.mass - original.mass) / original.mass
        if mass_error > self.validation_thresholds['mass_tolerance']:
            validation_result['errors'].append(f"Mass conservation violated: {mass_error:.3f}")
            validation_result['is_valid'] = False

        # 마찰계수 합리성 검사
        friction_ratio = converted.friction_coefficient / original.friction_coefficient
        if not (0.5 <= friction_ratio <= 2.0):
            validation_result['warnings'].append(f"Large friction change: {friction_ratio:.2f}x")

        # 에너지 보존 추정
        energy_conservation = self._estimate_energy_conservation(original, converted)
        validation_result['conservation_metrics']['energy'] = energy_conservation

        if energy_conservation < (1.0 - self.validation_thresholds['energy_conservation']):
            validation_result['warnings'].append(f"Potential energy loss: {1-energy_conservation:.3f}")

        # 전체 정확도 점수 계산
        validation_result['accuracy_score'] = self._calculate_accuracy_score(
            original, converted, validation_result
        )

        return validation_result

    def _estimate_energy_conservation(self, original: PhysicalProperties,
                                    converted: PhysicalProperties) -> float:
        """에너지 보존 추정"""
        # 간단한 추정: 질량과 탄성계수 기반
        mass_ratio = min(converted.mass, original.mass) / max(converted.mass, original.mass)
        restitution_ratio = min(converted.restitution, original.restitution) / \
                           max(converted.restitution, original.restitution)

        # 기하평균으로 종합 점수 계산
        conservation_score = np.sqrt(mass_ratio * restitution_ratio)
        return conservation_score

    def _calculate_accuracy_score(self, original: PhysicalProperties,
                                converted: PhysicalProperties,
                                validation_result: Dict[str, Any]) -> float:
        """정확도 점수 계산"""
        base_score = 1.0

        # 오류가 있으면 점수 감점
        error_count = len(validation_result['errors'])
        warning_count = len(validation_result['warnings'])

        base_score -= error_count * 0.3  # 오류당 30% 감점
        base_score -= warning_count * 0.1  # 경고당 10% 감점

        # 에너지 보존 점수 반영
        energy_score = validation_result['conservation_metrics'].get('energy', 1.0)
        base_score *= energy_score

        return max(base_score, 0.0)

    def estimate_accuracy(self, source_engine: PhysicsEngine,
                         target_engine: PhysicsEngine) -> float:
        """변환 정확도 추정"""
        if source_engine == target_engine:
            return 1.0

        # 엔진 간 호환성 매트릭스
        compatibility_matrix = {
            (PhysicsEngine.ISAAC_SIM, PhysicsEngine.GENESIS_AI): 0.85,
            (PhysicsEngine.GENESIS_AI, PhysicsEngine.ISAAC_SIM): 0.85,
            (PhysicsEngine.ISAAC_SIM, PhysicsEngine.PYBULLET): 0.70,
            (PhysicsEngine.PYBULLET, PhysicsEngine.ISAAC_SIM): 0.70,
            (PhysicsEngine.GENESIS_AI, PhysicsEngine.PYBULLET): 0.75,
            (PhysicsEngine.PYBULLET, PhysicsEngine.GENESIS_AI): 0.75,
        }

        return compatibility_matrix.get((source_engine, target_engine), 0.60)

    def _load_engine_profiles(self) -> Dict[PhysicsEngine, PhysicsEngineProfile]:
        """물리 엔진 프로필 로딩"""
        profiles = {}

        # Isaac Sim 프로필
        profiles[PhysicsEngine.ISAAC_SIM] = PhysicsEngineProfile(
            engine_type=PhysicsEngine.ISAAC_SIM,
            version="2023.1",
            supported_features=['soft_body', 'fluid', 'cloth', 'rigid_body'],
            precision_level='high',
            simulation_frequency=120.0,
            numerical_stability=0.9,
            convergence_threshold=1e-6,
            maximum_iterations=20,
            contact_model='penalty',
            friction_model='coulomb',
            collision_detection='continuous'
        )

        # Genesis AI 프로필
        profiles[PhysicsEngine.GENESIS_AI] = PhysicsEngineProfile(
            engine_type=PhysicsEngine.GENESIS_AI,
            version="0.3.3",
            supported_features=['rigid_body', 'soft_body', 'fluid'],
            precision_level='high',
            simulation_frequency=100.0,
            numerical_stability=0.95,
            convergence_threshold=1e-5,
            maximum_iterations=10,
            contact_model='impulse',
            friction_model='box',
            collision_detection='discrete'
        )

        return profiles

    def _initialize_conversion_matrices(self) -> Dict[str, Dict[str, float]]:
        """변환 매트릭스 초기화"""
        matrices = {}

        # Isaac Sim → Genesis AI
        matrices['isaac_sim_to_genesis_ai'] = PhysicsConstants.ISAAC_TO_GENESIS_SCALING.copy()

        # Genesis AI → Isaac Sim (역변환)
        matrices['genesis_ai_to_isaac_sim'] = {
            key: 1.0 / value for key, value in
            PhysicsConstants.ISAAC_TO_GENESIS_SCALING.items()
        }

        # 기타 엔진 간 변환 매트릭스는 필요에 따라 추가

        return matrices


# ============================================================================
# 궤적 데이터 변환 시스템
# ============================================================================

class TrajectoryProcessor:
    """궤적 데이터 처리 및 변환"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def resample_trajectory(self, trajectory: TrajectoryData,
                           target_frequency: float) -> TrajectoryData:
        """궤적 데이터 리샘플링"""
        from scipy import interpolate

        current_frequency = 1.0 / np.mean(np.diff(trajectory.timestamps))

        if abs(current_frequency - target_frequency) < 1e-3:
            return trajectory  # 이미 원하는 주파수

        self.logger.info(f"Resampling trajectory: {current_frequency:.1f}Hz → {target_frequency:.1f}Hz")

        # 새로운 시간 배열 생성
        duration = trajectory.timestamps[-1] - trajectory.timestamps[0]
        new_timestamps = np.linspace(
            trajectory.timestamps[0],
            trajectory.timestamps[-1],
            int(duration * target_frequency) + 1
        )

        # 관절 위치 보간
        interp_func = interpolate.interp1d(
            trajectory.timestamps,
            trajectory.joint_positions.T,
            kind='cubic',
            axis=1,
            fill_value='extrapolate'
        )
        new_joint_positions = interp_func(new_timestamps).T

        # 속도 계산 (중앙차분)
        new_joint_velocities = np.gradient(new_joint_positions, new_timestamps, axis=0)

        # 가속도 계산
        new_joint_accelerations = np.gradient(new_joint_velocities, new_timestamps, axis=0)

        return TrajectoryData(
            timestamps=new_timestamps,
            joint_positions=new_joint_positions,
            joint_velocities=new_joint_velocities,
            joint_accelerations=new_joint_accelerations,
            joint_torques=trajectory.joint_torques,  # 토크는 별도 처리 필요
            end_effector_poses=trajectory.end_effector_poses,
            end_effector_forces=trajectory.end_effector_forces,
            camera_images=trajectory.camera_images,
            depth_images=trajectory.depth_images,
            sampling_rate=target_frequency,
            coordinate_frame=trajectory.coordinate_frame
        )

    def smooth_trajectory(self, trajectory: TrajectoryData,
                         smoothing_factor: float = 0.1) -> TrajectoryData:
        """궤적 평활화"""
        from scipy import ndimage

        sigma = smoothing_factor * len(trajectory.timestamps) / 10.0

        smoothed_positions = ndimage.gaussian_filter1d(
            trajectory.joint_positions, sigma, axis=0
        )

        # 평활화된 위치에서 속도 재계산
        smoothed_velocities = np.gradient(smoothed_positions, trajectory.timestamps, axis=0)
        smoothed_accelerations = np.gradient(smoothed_velocities, trajectory.timestamps, axis=0)

        return TrajectoryData(
            timestamps=trajectory.timestamps,
            joint_positions=smoothed_positions,
            joint_velocities=smoothed_velocities,
            joint_accelerations=smoothed_accelerations,
            joint_torques=trajectory.joint_torques,
            end_effector_poses=trajectory.end_effector_poses,
            end_effector_forces=trajectory.end_effector_forces,
            camera_images=trajectory.camera_images,
            depth_images=trajectory.depth_images,
            sampling_rate=trajectory.sampling_rate,
            coordinate_frame=trajectory.coordinate_frame
        )

    def filter_outliers(self, trajectory: TrajectoryData,
                       threshold_std: float = 3.0) -> TrajectoryData:
        """이상값 필터링"""
        positions = trajectory.joint_positions.copy()

        # 각 관절별로 이상값 감지
        for joint_idx in range(positions.shape[1]):
            joint_data = positions[:, joint_idx]
            mean_val = np.mean(joint_data)
            std_val = np.std(joint_data)

            # 3σ 규칙 적용
            outlier_mask = np.abs(joint_data - mean_val) > threshold_std * std_val

            if np.any(outlier_mask):
                self.logger.warning(f"Detected {np.sum(outlier_mask)} outliers in joint {joint_idx}")

                # 보간으로 이상값 대체
                outlier_indices = np.where(outlier_mask)[0]
                good_indices = np.where(~outlier_mask)[0]

                if len(good_indices) > 0:
                    positions[outlier_indices, joint_idx] = np.interp(
                        outlier_indices, good_indices, joint_data[good_indices]
                    )

        # 속도와 가속도 재계산
        velocities = np.gradient(positions, trajectory.timestamps, axis=0)
        accelerations = np.gradient(velocities, trajectory.timestamps, axis=0)

        return TrajectoryData(
            timestamps=trajectory.timestamps,
            joint_positions=positions,
            joint_velocities=velocities,
            joint_accelerations=accelerations,
            joint_torques=trajectory.joint_torques,
            end_effector_poses=trajectory.end_effector_poses,
            end_effector_forces=trajectory.end_effector_forces,
            camera_images=trajectory.camera_images,
            depth_images=trajectory.depth_images,
            sampling_rate=trajectory.sampling_rate,
            coordinate_frame=trajectory.coordinate_frame
        )


# ============================================================================
# 단위 시스템 정규화
# ============================================================================

class UnitSystemNormalizer:
    """단위 시스템 정규화"""

    # SI 기본 단위 정의
    SI_UNITS = {
        'length': 'm',      # meter
        'mass': 'kg',       # kilogram
        'time': 's',        # second
        'angle': 'rad',     # radian
        'force': 'N',       # newton
        'torque': 'Nm',     # newton-meter
        'velocity': 'm/s',  # meter per second
        'angular_velocity': 'rad/s',  # radian per second
    }

    # 변환 팩터 (목표 단위 → SI 단위)
    CONVERSION_FACTORS = {
        # 길이
        'mm': 0.001, 'cm': 0.01, 'dm': 0.1, 'm': 1.0, 'km': 1000.0,
        'in': 0.0254, 'ft': 0.3048, 'yd': 0.9144,

        # 질량
        'g': 0.001, 'kg': 1.0, 'lb': 0.453592,

        # 각도
        'deg': np.pi / 180.0, 'rad': 1.0,

        # 힘
        'N': 1.0, 'kN': 1000.0, 'lbf': 4.44822,

        # 토크
        'Nm': 1.0, 'kNm': 1000.0, 'lbf_ft': 1.35582,
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def normalize_physical_properties(self, props: PhysicalProperties,
                                    source_units: Dict[str, str]) -> PhysicalProperties:
        """물리 속성을 SI 단위로 정규화"""
        normalized_props = PhysicalProperties(
            mass=self._convert_to_si(props.mass, source_units.get('mass', 'kg'), 'kg'),
            friction_coefficient=props.friction_coefficient,  # 무차원
            restitution=props.restitution,  # 무차원
            linear_damping=props.linear_damping,  # 단위 의존적이므로 별도 처리 필요
            angular_damping=props.angular_damping,
            inertia_tensor=props.inertia_tensor,
            center_of_mass=self._convert_vector_to_si(
                props.center_of_mass, source_units.get('length', 'm'), 'm'
            ) if props.center_of_mass is not None else None,
            contact_stiffness=self._convert_to_si(
                props.contact_stiffness, source_units.get('force', 'N'), 'N'
            ) if props.contact_stiffness is not None else None,
            contact_damping=props.contact_damping,  # 복잡한 단위이므로 별도 처리
            material_type=props.material_type,
            surface_roughness=self._convert_to_si(
                props.surface_roughness, source_units.get('length', 'm'), 'm'
            ) if props.surface_roughness is not None else None,
            density=self._convert_to_si(
                props.density, f"{source_units.get('mass', 'kg')}/{source_units.get('length', 'm')}3", 'kg/m3'
            ) if props.density is not None else None,
        )

        return normalized_props

    def normalize_robot_configuration(self, config: RobotConfiguration,
                                    source_units: Dict[str, str]) -> RobotConfiguration:
        """로봇 구성을 SI 단위로 정규화"""
        length_factor = self.CONVERSION_FACTORS.get(source_units.get('length', 'm'), 1.0)
        mass_factor = self.CONVERSION_FACTORS.get(source_units.get('mass', 'kg'), 1.0)
        force_factor = self.CONVERSION_FACTORS.get(source_units.get('force', 'N'), 1.0)

        # 관절 제한값 변환 (각도인 경우)
        angle_factor = self.CONVERSION_FACTORS.get(source_units.get('angle', 'rad'), 1.0)
        normalized_limits = {}
        for joint_name, (min_val, max_val) in config.joint_limits.items():
            normalized_limits[joint_name] = (min_val * angle_factor, max_val * angle_factor)

        normalized_config = RobotConfiguration(
            robot_type=config.robot_type,
            joint_count=config.joint_count,
            joint_names=config.joint_names.copy(),
            joint_types=config.joint_types.copy(),
            joint_limits=normalized_limits,
            base_pose=config.base_pose.copy(),  # 변환 행렬은 별도 처리
            end_effector_offset=config.end_effector_offset * length_factor,
            link_lengths=[length * length_factor for length in config.link_lengths],
            link_masses=[mass * mass_factor for mass in config.link_masses],
            joint_stiffness=config.joint_stiffness.copy(),  # 복잡한 단위
            joint_damping=config.joint_damping.copy(),      # 복잡한 단위
            max_joint_velocities=[vel * angle_factor for vel in config.max_joint_velocities],
            max_joint_torques=[torque * force_factor * length_factor for torque in config.max_joint_torques]
        )

        return normalized_config

    def _convert_to_si(self, value: float, source_unit: str, target_unit: str) -> float:
        """단일 값을 SI 단위로 변환"""
        if value is None:
            return None

        source_factor = self.CONVERSION_FACTORS.get(source_unit, 1.0)
        target_factor = self.CONVERSION_FACTORS.get(target_unit, 1.0)

        # source_unit → SI → target_unit
        si_value = value * source_factor
        target_value = si_value / target_factor

        return target_value

    def _convert_vector_to_si(self, vector: np.ndarray, source_unit: str, target_unit: str) -> np.ndarray:
        """벡터를 SI 단위로 변환"""
        if vector is None:
            return None

        factor = self._convert_to_si(1.0, source_unit, target_unit)
        return vector * factor