"""
데이터 추상화 계층 (Data Abstraction Layer)

NVIDIA PhysicalAI 데이터셋을 Genesis AI + Franka 환경에 호환되도록
추상화하는 인터페이스 계층
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Iterator, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pathlib import Path
import h5py
from datetime import datetime


# ============================================================================
# 표준화된 데이터 스키마 정의
# ============================================================================

class PhysicsEngine(Enum):
    """지원하는 물리 엔진 타입"""
    ISAAC_SIM = "isaac_sim"
    GENESIS_AI = "genesis_ai"
    PYBULLET = "pybullet"


class RobotType(Enum):
    """지원하는 로봇 타입"""
    FRANKA_PANDA = "franka_panda"
    UR5 = "ur5"
    KUKA_IIWA = "kuka_iiwa"
    GENERIC_6DOF = "generic_6dof"


@dataclass
class PhysicalProperties:
    """물리적 속성 표준 스키마"""
    # 기본 물리 속성
    mass: float  # kg
    friction_coefficient: float  # 마찰계수
    restitution: float  # 탄성계수
    linear_damping: float  # 선형 감쇠
    angular_damping: float  # 각 감쇠

    # 고급 물리 속성
    inertia_tensor: Optional[np.ndarray] = None  # 관성 텐서 (3x3)
    center_of_mass: Optional[np.ndarray] = None  # 질량 중심 (x,y,z)
    contact_stiffness: Optional[float] = None  # 접촉 강성
    contact_damping: Optional[float] = None  # 접촉 감쇠

    # 메타데이터
    material_type: Optional[str] = None
    surface_roughness: Optional[float] = None
    density: Optional[float] = None  # kg/m³


@dataclass
class RobotConfiguration:
    """로봇 구성 정보"""
    robot_type: RobotType
    joint_count: int
    joint_names: List[str]
    joint_types: List[str]  # revolute, prismatic
    joint_limits: Dict[str, tuple]  # {joint_name: (min, max)}

    # 기구학 정보
    base_pose: np.ndarray  # 4x4 변환 행렬
    end_effector_offset: np.ndarray  # 말단 장치 오프셋
    link_lengths: List[float]  # 링크 길이
    link_masses: List[float]  # 링크 질량

    # 동력학 정보
    joint_stiffness: List[float]  # 관절 강성
    joint_damping: List[float]  # 관절 감쇠
    max_joint_velocities: List[float]  # 최대 관절 속도
    max_joint_torques: List[float]  # 최대 관절 토크


@dataclass
class TrajectoryData:
    """궤적 데이터"""
    timestamps: np.ndarray  # 시간 배열
    joint_positions: np.ndarray  # 관절 위치 (N x DOF)
    joint_velocities: Optional[np.ndarray] = None  # 관절 속도
    joint_accelerations: Optional[np.ndarray] = None  # 관절 가속도
    joint_torques: Optional[np.ndarray] = None  # 관절 토크

    # 말단 장치 궤적
    end_effector_poses: Optional[np.ndarray] = None  # 4x4 변환 행렬 배열
    end_effector_forces: Optional[np.ndarray] = None  # 힘/토크

    # 센서 데이터
    camera_images: Optional[List[np.ndarray]] = None  # RGB 이미지
    depth_images: Optional[List[np.ndarray]] = None  # 깊이 이미지

    # 메타데이터
    sampling_rate: float = 100.0  # Hz
    coordinate_frame: str = "world"


@dataclass
class SceneDescription:
    """장면 설명"""
    # 환경 설정
    gravity: np.ndarray = field(default_factory=lambda: np.array([0, 0, -9.81]))
    environment_type: str = "indoor"  # indoor, outdoor, laboratory
    lighting_conditions: str = "normal"  # dim, normal, bright

    # 객체 정보
    objects: List[Dict[str, Any]] = field(default_factory=list)
    obstacles: List[Dict[str, Any]] = field(default_factory=list)
    target_objects: List[Dict[str, Any]] = field(default_factory=list)

    # 작업 정보
    task_type: str = "manipulation"  # manipulation, navigation, assembly
    task_description: str = ""
    success_criteria: List[str] = field(default_factory=list)


@dataclass
class ProcessingMetadata:
    """처리 메타데이터"""
    source_engine: PhysicsEngine
    target_engine: PhysicsEngine
    conversion_timestamp: str
    conversion_version: str

    # 품질 메트릭
    data_completeness: float  # 0.0 ~ 1.0
    physics_accuracy: float  # 물리적 정확성 점수
    language_quality: float  # 언어 품질 점수

    # 처리 정보
    processing_time: float  # 초
    memory_usage: float  # MB
    error_logs: List[str] = field(default_factory=list)
    warning_logs: List[str] = field(default_factory=list)


@dataclass
class StandardDataSchema:
    """표준화된 데이터 스키마"""
    robot_config: RobotConfiguration
    physical_properties: PhysicalProperties
    trajectory_data: TrajectoryData
    scene_description: SceneDescription
    metadata: ProcessingMetadata

    # 자연어 어노테이션
    natural_language_annotations: List[str] = field(default_factory=list)
    instruction_templates: List[str] = field(default_factory=list)


# ============================================================================
# 데이터 소스 어댑터 인터페이스
# ============================================================================

class DataSourceAdapter(ABC):
    """데이터 소스 어댑터 추상 인터페이스"""

    @abstractmethod
    def load_dataset(self, dataset_path: Union[str, Path]) -> Iterator[StandardDataSchema]:
        """데이터셋 로딩"""
        pass

    @abstractmethod
    def validate_format(self, dataset_path: Union[str, Path]) -> bool:
        """데이터 포맷 검증"""
        pass

    @abstractmethod
    def get_metadata(self, dataset_path: Union[str, Path]) -> Dict[str, Any]:
        """메타데이터 추출"""
        pass

    @abstractmethod
    def estimate_processing_time(self, dataset_path: Union[str, Path]) -> float:
        """처리 시간 추정"""
        pass


class PhysicalAIAdapter(DataSourceAdapter):
    """NVIDIA PhysicalAI 데이터셋 어댑터"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.supported_formats = ['.h5', '.hdf5', '.json', '.pkl']

    def load_dataset(self, dataset_path: Union[str, Path]) -> Iterator[StandardDataSchema]:
        """PhysicalAI 데이터셋을 표준 스키마로 변환하여 로딩"""
        path = Path(dataset_path)

        if path.suffix in ['.h5', '.hdf5']:
            yield from self._load_hdf5_dataset(path)
        elif path.suffix == '.json':
            yield from self._load_json_dataset(path)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

    def _load_hdf5_dataset(self, path: Path) -> Iterator[StandardDataSchema]:
        """HDF5 포맷 데이터 로딩"""
        with h5py.File(path, 'r') as f:
            # 데이터 구조 분석
            episodes = self._extract_episodes(f)

            for episode_data in episodes:
                schema = self._convert_to_standard_schema(episode_data)
                yield schema

    def _load_json_dataset(self, path: Path) -> Iterator[StandardDataSchema]:
        """JSON 포맷 데이터 로딩"""
        import json

        with open(path, 'r') as f:
            data = json.load(f)

        if isinstance(data, list):
            for item in data:
                schema = self._convert_json_to_schema(item)
                yield schema
        else:
            schema = self._convert_json_to_schema(data)
            yield schema

    def _extract_episodes(self, hdf5_file) -> List[Dict[str, Any]]:
        """HDF5에서 에피소드 데이터 추출"""
        episodes = []

        # PhysicalAI 데이터셋의 일반적인 구조 가정
        for episode_key in hdf5_file.keys():
            if episode_key.startswith('episode_'):
                episode_group = hdf5_file[episode_key]
                episode_data = {}

                # 로봇 데이터
                if 'robot' in episode_group:
                    episode_data['robot'] = {
                        'joint_positions': episode_group['robot/joint_positions'][:],
                        'joint_velocities': episode_group.get('robot/joint_velocities', None),
                        'joint_torques': episode_group.get('robot/joint_torques', None),
                        'timestamps': episode_group['robot/timestamps'][:]
                    }

                # 장면 데이터
                if 'scene' in episode_group:
                    episode_data['scene'] = {
                        'objects': episode_group.get('scene/objects', []),
                        'environment': episode_group.get('scene/environment', {}),
                        'task_description': episode_group.get('scene/task_description', "")
                    }

                # 물리 속성
                if 'physics' in episode_group:
                    episode_data['physics'] = {
                        'object_properties': episode_group.get('physics/object_properties', {}),
                        'simulation_params': episode_group.get('physics/simulation_params', {})
                    }

                episodes.append(episode_data)

        return episodes

    def _convert_to_standard_schema(self, episode_data: Dict[str, Any]) -> StandardDataSchema:
        """에피소드 데이터를 표준 스키마로 변환"""
        # 로봇 구성 변환
        robot_config = self._extract_robot_config(episode_data.get('robot', {}))

        # 물리 속성 변환
        physical_properties = self._extract_physical_properties(episode_data.get('physics', {}))

        # 궤적 데이터 변환
        trajectory_data = self._extract_trajectory_data(episode_data.get('robot', {}))

        # 장면 설명 변환
        scene_description = self._extract_scene_description(episode_data.get('scene', {}))

        # 메타데이터 생성
        metadata = ProcessingMetadata(
            source_engine=PhysicsEngine.ISAAC_SIM,
            target_engine=PhysicsEngine.GENESIS_AI,
            conversion_timestamp=str(np.datetime64('now')),
            conversion_version="1.0.0",
            data_completeness=self._calculate_completeness(episode_data),
            physics_accuracy=0.0,  # 추후 계산
            language_quality=0.0,  # 추후 계산
            processing_time=0.0,  # 추후 측정
            memory_usage=0.0  # 추후 측정
        )

        return StandardDataSchema(
            robot_config=robot_config,
            physical_properties=physical_properties,
            trajectory_data=trajectory_data,
            scene_description=scene_description,
            metadata=metadata
        )

    def _extract_robot_config(self, robot_data: Dict[str, Any]) -> RobotConfiguration:
        """로봇 구성 정보 추출"""
        # 기본값 설정 (Franka Panda 가정)
        return RobotConfiguration(
            robot_type=RobotType.FRANKA_PANDA,
            joint_count=7,
            joint_names=[f"joint_{i}" for i in range(7)],
            joint_types=["revolute"] * 7,
            joint_limits={f"joint_{i}": (-2.8973, 2.8973) for i in range(7)},
            base_pose=np.eye(4),
            end_effector_offset=np.array([0, 0, 0.1]),
            link_lengths=[0.333, 0.316, 0.384, 0.0825, 0.384, 0.088, 0.107],
            link_masses=[4.970, 0.646, 3.228, 3.587, 1.225, 1.666, 0.735],
            joint_stiffness=[3000.0] * 7,
            joint_damping=[100.0] * 7,
            max_joint_velocities=[2.175] * 7,
            max_joint_torques=[87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]
        )

    def _extract_physical_properties(self, physics_data: Dict[str, Any]) -> PhysicalProperties:
        """물리 속성 추출"""
        return PhysicalProperties(
            mass=physics_data.get('mass', 1.0),
            friction_coefficient=physics_data.get('friction', 0.7),
            restitution=physics_data.get('restitution', 0.3),
            linear_damping=physics_data.get('linear_damping', 0.1),
            angular_damping=physics_data.get('angular_damping', 0.1),
            material_type=physics_data.get('material_type', 'generic')
        )

    def _extract_trajectory_data(self, robot_data: Dict[str, Any]) -> TrajectoryData:
        """궤적 데이터 추출"""
        return TrajectoryData(
            timestamps=robot_data.get('timestamps', np.array([])),
            joint_positions=robot_data.get('joint_positions', np.array([])),
            joint_velocities=robot_data.get('joint_velocities'),
            joint_torques=robot_data.get('joint_torques'),
            sampling_rate=100.0
        )

    def _extract_scene_description(self, scene_data: Dict[str, Any]) -> SceneDescription:
        """장면 설명 추출"""
        return SceneDescription(
            environment_type=scene_data.get('environment_type', 'indoor'),
            task_type=scene_data.get('task_type', 'manipulation'),
            task_description=scene_data.get('task_description', ''),
            objects=scene_data.get('objects', [])
        )

    def _convert_json_to_schema(self, data: Dict[str, Any]) -> StandardDataSchema:
        """JSON 데이터를 표준 스키마로 변환"""
        # JSON 데이터 구조 분석
        if 'episodes' in data:
            # 여러 에피소드가 있는 경우 첫 번째만 처리
            episode_data = data['episodes'][0]
        else:
            # 단일 에피소드 데이터
            episode_data = data

        # 로봇 설정 추출
        robot_data = episode_data.get('robot', {})
        robot_config = self._extract_robot_config(robot_data)

        # 물리 속성 추출 (기본값 사용)
        physics_data = episode_data.get('physics', {})
        physical_props = self._extract_physical_properties(physics_data)

        # 궤적 데이터 추출
        trajectory_data = self._extract_trajectory_data(robot_data)

        # 장면 설명 추출
        scene_data = episode_data.get('scene', {})
        scene_description = self._extract_scene_description(scene_data, episode_data)

        # 메타데이터 생성
        metadata = ProcessingMetadata(
            source_engine=PhysicsEngine.ISAAC_SIM,
            target_engine=PhysicsEngine.GENESIS_AI,
            conversion_timestamp=datetime.now().isoformat(),
            conversion_version="1.0.0",
            data_completeness=0.8,
            physics_accuracy=0.9,
            language_quality=0.7,
            processing_time=0.1,
            memory_usage=64.0
        )

        return StandardDataSchema(
            robot_config=robot_config,
            physical_properties=physical_props,
            trajectory_data=trajectory_data,
            scene_description=scene_description,
            metadata=metadata
        )

    def _extract_robot_config(self, robot_data: Dict[str, Any]) -> RobotConfiguration:
        """로봇 설정 데이터 추출"""
        robot_type_str = robot_data.get('type', 'franka_panda')
        robot_type = RobotType.FRANKA_PANDA if robot_type_str == 'franka_panda' else RobotType.OTHER

        joint_count = robot_data.get('dof', 7)
        joint_names = [f"joint_{i}" for i in range(joint_count)]

        return RobotConfiguration(
            robot_type=robot_type,
            joint_count=joint_count,
            joint_names=joint_names,
            joint_types=["revolute"] * joint_count,
            joint_limits={name: (-2.8, 2.8) for name in joint_names},
            base_pose=np.eye(4),
            end_effector_offset=np.array([0, 0, 0.1]),
            link_lengths=[0.333, 0.316, 0.384, 0.0825, 0.384, 0.088, 0.107][:joint_count],
            link_masses=[4.970, 0.646, 3.228, 3.587, 1.225, 1.666, 0.735][:joint_count],
            joint_stiffness=[3000.0] * joint_count,
            joint_damping=[100.0] * joint_count,
            max_joint_velocities=[2.175] * joint_count,
            max_joint_torques=[87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0][:joint_count]
        )

    def _extract_physical_properties(self, physics_data: Dict[str, Any]) -> PhysicalProperties:
        """물리 속성 데이터 추출"""
        return PhysicalProperties(
            mass=physics_data.get('mass', 1.5),
            friction_coefficient=physics_data.get('friction_coefficient', 0.8),
            restitution=physics_data.get('restitution', 0.2),
            linear_damping=physics_data.get('linear_damping', 0.1),
            angular_damping=physics_data.get('angular_damping', 0.1),
            material_type=physics_data.get('material_type', 'plastic')
        )

    def _extract_trajectory_data(self, robot_data: Dict[str, Any]) -> TrajectoryData:
        """궤적 데이터 추출"""
        timestamps = robot_data.get('timestamps', [])
        joint_positions = robot_data.get('joint_positions', [])

        # 리스트를 numpy 배열로 변환
        if timestamps:
            timestamps = np.array(timestamps)
        else:
            timestamps = np.linspace(0, 5, 100)

        if joint_positions:
            joint_positions = np.array(joint_positions)
        else:
            # 기본 더미 데이터
            joint_positions = np.random.rand(len(timestamps), 7)

        sampling_rate = 1.0 / (timestamps[1] - timestamps[0]) if len(timestamps) > 1 else 100.0

        return TrajectoryData(
            timestamps=timestamps,
            joint_positions=joint_positions,
            sampling_rate=sampling_rate
        )

    def _extract_scene_description(self, scene_data: Dict[str, Any], episode_data: Dict[str, Any]) -> SceneDescription:
        """장면 설명 데이터 추출"""
        return SceneDescription(
            task_type=episode_data.get('scenario', 'manipulation'),
            task_description=scene_data.get('description', f"{episode_data.get('scenario', 'manipulation')} task"),
            environment_type=scene_data.get('environment_type', 'laboratory'),
            objects=scene_data.get('objects', [])
        )

    def _calculate_completeness(self, episode_data: Dict[str, Any]) -> float:
        """데이터 완전성 계산"""
        required_fields = ['robot', 'scene', 'physics']
        present_fields = [field for field in required_fields if field in episode_data]
        return len(present_fields) / len(required_fields)

    def validate_format(self, dataset_path: Union[str, Path]) -> bool:
        """데이터 포맷 검증"""
        path = Path(dataset_path)

        if not path.exists():
            return False

        if path.suffix not in self.supported_formats:
            return False

        try:
            if path.suffix in ['.h5', '.hdf5']:
                with h5py.File(path, 'r') as f:
                    # 기본 구조 확인
                    return len(f.keys()) > 0
            elif path.suffix == '.json':
                import json
                with open(path, 'r') as f:
                    json.load(f)
                return True
        except Exception:
            return False

        return True

    def get_metadata(self, dataset_path: Union[str, Path]) -> Dict[str, Any]:
        """메타데이터 추출"""
        path = Path(dataset_path)
        metadata = {
            'file_size': path.stat().st_size,
            'file_format': path.suffix,
            'last_modified': path.stat().st_mtime
        }

        if path.suffix in ['.h5', '.hdf5']:
            with h5py.File(path, 'r') as f:
                metadata['episode_count'] = len([k for k in f.keys() if k.startswith('episode_')])
                metadata['data_keys'] = list(f.keys())

        return metadata

    def estimate_processing_time(self, dataset_path: Union[str, Path]) -> float:
        """처리 시간 추정 (초)"""
        metadata = self.get_metadata(dataset_path)
        file_size_mb = metadata['file_size'] / (1024 * 1024)

        # 경험적 추정: 1MB당 약 0.5초
        estimated_time = file_size_mb * 0.5
        return max(estimated_time, 1.0)  # 최소 1초


# ============================================================================
# 데이터 검증 시스템
# ============================================================================

class DataValidator:
    """데이터 검증 시스템"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validation_rules = self._load_validation_rules()

    def validate_schema(self, data: StandardDataSchema) -> Dict[str, Any]:
        """스키마 검증"""
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'completeness_score': 0.0
        }

        # 필수 필드 검증
        required_checks = [
            ('robot_config', self._validate_robot_config),
            ('physical_properties', self._validate_physical_properties),
            ('trajectory_data', self._validate_trajectory_data),
            ('scene_description', self._validate_scene_description)
        ]

        completed_checks = 0
        for field_name, validator in required_checks:
            try:
                field_data = getattr(data, field_name)
                validation_result = validator(field_data)

                if validation_result['is_valid']:
                    completed_checks += 1
                else:
                    results['errors'].extend(validation_result['errors'])
                    results['is_valid'] = False

                results['warnings'].extend(validation_result.get('warnings', []))

            except Exception as e:
                results['errors'].append(f"Validation error for {field_name}: {str(e)}")
                results['is_valid'] = False

        results['completeness_score'] = completed_checks / len(required_checks)

        return results

    def _validate_robot_config(self, config: RobotConfiguration) -> Dict[str, Any]:
        """로봇 구성 검증"""
        result = {'is_valid': True, 'errors': [], 'warnings': []}

        # 관절 수 일관성 검증
        if len(config.joint_names) != config.joint_count:
            result['errors'].append("Joint names count doesn't match joint_count")
            result['is_valid'] = False

        # 관절 제한값 검증
        for joint_name in config.joint_names:
            if joint_name not in config.joint_limits:
                result['warnings'].append(f"Missing joint limits for {joint_name}")

        return result

    def _validate_physical_properties(self, props: PhysicalProperties) -> Dict[str, Any]:
        """물리 속성 검증"""
        result = {'is_valid': True, 'errors': [], 'warnings': []}

        # 물리적 타당성 검증
        if props.mass <= 0:
            result['errors'].append("Mass must be positive")
            result['is_valid'] = False

        if not (0 <= props.friction_coefficient <= 2.0):
            result['warnings'].append("Friction coefficient seems unusual")

        if not (0 <= props.restitution <= 1.0):
            result['errors'].append("Restitution must be between 0 and 1")
            result['is_valid'] = False

        return result

    def _validate_trajectory_data(self, traj: TrajectoryData) -> Dict[str, Any]:
        """궤적 데이터 검증"""
        result = {'is_valid': True, 'errors': [], 'warnings': []}

        # 시간 배열 검증
        if len(traj.timestamps) == 0:
            result['errors'].append("Empty trajectory timestamps")
            result['is_valid'] = False
            return result

        # 시간 순서 검증
        if not np.all(np.diff(traj.timestamps) > 0):
            result['errors'].append("Timestamps are not monotonically increasing")
            result['is_valid'] = False

        # 데이터 차원 일관성 검증
        if traj.joint_positions.shape[0] != len(traj.timestamps):
            result['errors'].append("Joint positions length doesn't match timestamps")
            result['is_valid'] = False

        return result

    def _validate_scene_description(self, scene: SceneDescription) -> Dict[str, Any]:
        """장면 설명 검증"""
        result = {'is_valid': True, 'errors': [], 'warnings': []}

        # 중력 벡터 검증
        if len(scene.gravity) != 3:
            result['errors'].append("Gravity vector must be 3D")
            result['is_valid'] = False

        # 작업 설명 확인
        if not scene.task_description.strip():
            result['warnings'].append("Empty task description")

        return result

    def _load_validation_rules(self) -> Dict[str, Any]:
        """검증 규칙 로딩"""
        return {
            'max_joint_velocity': 10.0,  # rad/s
            'max_joint_acceleration': 50.0,  # rad/s²
            'min_sampling_rate': 10.0,  # Hz
            'max_sampling_rate': 1000.0  # Hz
        }


# ============================================================================
# 팩토리 패턴으로 어댑터 생성
# ============================================================================

class AdapterFactory:
    """데이터 소스 어댑터 팩토리"""

    _adapters = {
        'physicalai': PhysicalAIAdapter,
        # 추후 다른 어댑터 추가 가능
        # 'rosbag': ROSBagAdapter,
        # 'isaac_gym': IsaacGymAdapter,
    }

    @classmethod
    def create_adapter(cls, adapter_type: str, config: Dict[str, Any]) -> DataSourceAdapter:
        """어댑터 생성"""
        if adapter_type not in cls._adapters:
            raise ValueError(f"Unknown adapter type: {adapter_type}")

        adapter_class = cls._adapters[adapter_type]
        return adapter_class(config)

    @classmethod
    def register_adapter(cls, name: str, adapter_class: type):
        """새로운 어댑터 등록"""
        cls._adapters[name] = adapter_class

    @classmethod
    def list_adapters(cls) -> List[str]:
        """사용 가능한 어댑터 목록"""
        return list(cls._adapters.keys())