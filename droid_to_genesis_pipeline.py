#!/usr/bin/env python3
"""
DROID → Genesis AI + Franka 변환 파이프라인
공개 데이터셋을 Genesis AI 환경에서 Franka Panda 로봇에 맞게 변환
"""

import os
import json
import h5py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from enum import Enum
import time
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversionStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class RobotType(Enum):
    FRANKA_PANDA = "franka_panda"
    XARM = "xarm"
    ALLEGRO_HAND = "allegro_hand"
    UNKNOWN = "unknown"

@dataclass
class DroidEpisode:
    """DROID 에피소드 데이터 구조"""
    episode_id: str
    robot_type: str
    language_instruction: str
    joint_positions: np.ndarray
    end_effector_poses: np.ndarray
    gripper_states: np.ndarray
    timestamps: np.ndarray
    objects: List[Dict[str, Any]]
    scene_info: Dict[str, Any]

@dataclass
class GenesisEpisode:
    """Genesis AI 호환 에피소드 데이터 구조"""
    episode_id: str
    robot_config: Dict[str, Any]
    trajectory: Dict[str, np.ndarray]
    physics_properties: Dict[str, Any]
    language_annotations: List[str]
    scene_description: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class ConversionMetrics:
    """변환 품질 메트릭"""
    total_episodes: int = 0
    successful_conversions: int = 0
    failed_conversions: int = 0
    avg_trajectory_length: float = 0.0
    avg_processing_time: float = 0.0
    coordinate_transform_accuracy: float = 0.0
    kinematic_consistency: float = 0.0

class CoordinateTransformer:
    """좌표계 변환기 (DROID → Genesis AI)"""

    def __init__(self):
        """변환기 초기화"""
        # DROID는 ROS 좌표계, Genesis AI는 자체 좌표계 사용
        # ROS: x(forward), y(left), z(up)
        # Genesis AI: x(right), y(forward), z(up)
        self.transform_matrix = np.array([
            [0, 1, 0, 0],  # Genesis X = ROS Y
            [1, 0, 0, 0],  # Genesis Y = ROS X
            [0, 0, 1, 0],  # Genesis Z = ROS Z
            [0, 0, 0, 1]
        ])

    def transform_position(self, position: np.ndarray) -> np.ndarray:
        """위치 좌표 변환"""
        if position.shape[-1] == 3:
            # 3D 위치를 homogeneous 좌표로 변환
            ones = np.ones((*position.shape[:-1], 1))
            pos_homo = np.concatenate([position, ones], axis=-1)
            transformed = pos_homo @ self.transform_matrix.T
            return transformed[..., :3]
        return position

    def transform_orientation(self, quaternion: np.ndarray) -> np.ndarray:
        """방향 (쿼터니온) 변환"""
        if quaternion.shape[-1] == 4:
            # 쿼터니온을 회전 행렬로 변환하여 좌표계 변환 적용
            rotation = R.from_quat(quaternion)
            rotation_matrix = rotation.as_matrix()

            # 좌표계 변환 적용
            transform_rot = self.transform_matrix[:3, :3]
            transformed_matrix = transform_rot @ rotation_matrix @ transform_rot.T

            # 다시 쿼터니온으로 변환
            transformed_rotation = R.from_matrix(transformed_matrix)
            return transformed_rotation.as_quat()
        return quaternion

    def transform_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """전체 궤적 변환"""
        if trajectory.shape[-1] >= 6:  # position + orientation
            positions = trajectory[..., :3]
            orientations = trajectory[..., 3:7] if trajectory.shape[-1] >= 7 else None

            transformed_positions = self.transform_position(positions)

            if orientations is not None:
                transformed_orientations = self.transform_orientation(orientations)
                return np.concatenate([transformed_positions, transformed_orientations], axis=-1)
            else:
                return transformed_positions
        return trajectory

class FrankaKinematicMapper:
    """Franka Panda 키네마틱 매핑"""

    def __init__(self):
        """매퍼 초기화"""
        # Franka Panda DH 파라미터 (실제 값)
        self.dh_params = {
            'a': [0, 0, 0, 0.0825, -0.0825, 0, 0.088],
            'd': [0.333, 0, 0.316, 0, 0.384, 0, 0.107],
            'alpha': [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2],
            'joint_limits': [
                (-2.8973, 2.8973),  # Joint 1
                (-1.7628, 1.7628),  # Joint 2
                (-2.8973, 2.8973),  # Joint 3
                (-3.0718, -0.0698), # Joint 4
                (-2.8973, 2.8973),  # Joint 5
                (-0.0175, 3.7525),  # Joint 6
                (-2.8973, 2.8973)   # Joint 7
            ]
        }

    def validate_joint_limits(self, joint_positions: np.ndarray) -> Tuple[bool, List[str]]:
        """관절 한계 검증"""
        violations = []

        for i, (pos, (min_limit, max_limit)) in enumerate(zip(joint_positions.T, self.dh_params['joint_limits'])):
            out_of_bounds = (pos < min_limit) | (pos > max_limit)
            if np.any(out_of_bounds):
                violations.append(f"Joint {i+1}: {np.sum(out_of_bounds)} violations")

        is_valid = len(violations) == 0
        return is_valid, violations

    def clamp_joint_positions(self, joint_positions: np.ndarray) -> np.ndarray:
        """관절 위치를 허용 범위로 제한"""
        clamped = joint_positions.copy()

        for i, (min_limit, max_limit) in enumerate(self.dh_params['joint_limits']):
            clamped[:, i] = np.clip(clamped[:, i], min_limit, max_limit)

        return clamped

    def compute_forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        """순기구학 계산 (간소화된 구현)"""
        # 실제 구현에서는 정확한 DH 변환 사용
        # 여기서는 시뮬레이션을 위한 근사값 사용
        batch_size = joint_positions.shape[0]

        # 기본 엔드 이펙터 위치 (대략적)
        base_position = np.array([0.5, 0.0, 0.5])  # Genesis AI 좌표계

        # 관절 각도에 따른 변위 계산 (간소화)
        displacement = np.zeros((batch_size, 3))
        displacement[:, 0] = 0.2 * np.sin(joint_positions[:, 0])  # X
        displacement[:, 1] = 0.3 * np.cos(joint_positions[:, 1])  # Y
        displacement[:, 2] = 0.1 * joint_positions[:, 2]          # Z

        end_effector_positions = base_position + displacement

        # 방향은 기본값 사용 (단위 쿼터니온)
        orientations = np.tile([0, 0, 0, 1], (batch_size, 1))

        return np.concatenate([end_effector_positions, orientations], axis=1)

class PhysicsPropertyMapper:
    """물리 속성 매핑 (DROID → Genesis AI)"""

    def __init__(self):
        """매퍼 초기화"""
        self.material_mapping = {
            'plastic': {
                'density': 1200,  # kg/m³
                'friction': 0.4,
                'restitution': 0.2,
                'stiffness': 1e6
            },
            'metal': {
                'density': 7800,
                'friction': 0.6,
                'restitution': 0.1,
                'stiffness': 2e11
            },
            'wood': {
                'density': 600,
                'friction': 0.5,
                'restitution': 0.3,
                'stiffness': 1e10
            },
            'rubber': {
                'density': 1500,
                'friction': 0.8,
                'restitution': 0.9,
                'stiffness': 1e7
            },
            'default': {
                'density': 1000,
                'friction': 0.5,
                'restitution': 0.3,
                'stiffness': 1e8
            }
        }

    def map_object_properties(self, droid_object: Dict[str, Any]) -> Dict[str, Any]:
        """DROID 객체 속성을 Genesis AI 형식으로 매핑"""
        material = droid_object.get('material', 'default')
        mass = droid_object.get('mass', 0.1)

        # 재료별 기본 속성 가져오기
        base_properties = self.material_mapping.get(material, self.material_mapping['default'])

        # Genesis AI 물리 속성 구성
        genesis_properties = {
            'mass': mass,
            'density': base_properties['density'],
            'friction_coefficient': base_properties['friction'],
            'restitution_coefficient': base_properties['restitution'],
            'young_modulus': base_properties['stiffness'],
            'material_type': material,
            'collision_geometry': droid_object.get('geometry', 'box'),
            'visual_properties': {
                'color': droid_object.get('color', [0.5, 0.5, 0.5]),
                'texture': droid_object.get('texture', 'default')
            }
        }

        return genesis_properties

class LanguageProcessor:
    """자연어 명령 처리기"""

    def __init__(self):
        """프로세서 초기화"""
        self.action_keywords = {
            'pick': ['pick', 'grab', 'grasp', 'take', 'lift'],
            'place': ['place', 'put', 'set', 'drop', 'position'],
            'move': ['move', 'transport', 'carry', 'transfer'],
            'push': ['push', 'slide', 'shove'],
            'pull': ['pull', 'drag', 'draw']
        }

        self.object_keywords = {
            'container': ['bin', 'box', 'container', 'basket'],
            'tool': ['hammer', 'screwdriver', 'wrench'],
            'block': ['block', 'cube', 'brick'],
            'cup': ['cup', 'mug', 'glass', 'bottle']
        }

    def extract_action_intent(self, instruction: str) -> str:
        """동작 의도 추출"""
        instruction_lower = instruction.lower()

        for action, keywords in self.action_keywords.items():
            if any(keyword in instruction_lower for keyword in keywords):
                return action

        return 'unknown'

    def extract_objects(self, instruction: str) -> List[str]:
        """객체 추출"""
        instruction_lower = instruction.lower()
        extracted_objects = []

        for obj_type, keywords in self.object_keywords.items():
            if any(keyword in instruction_lower for keyword in keywords):
                extracted_objects.append(obj_type)

        return extracted_objects

    def normalize_instruction(self, instruction: str) -> str:
        """명령어 정규화"""
        # 기본 정규화
        normalized = instruction.strip().lower()

        # 특수 문자 제거
        import re
        normalized = re.sub(r'[^\w\s]', '', normalized)

        # 중복 공백 제거
        normalized = re.sub(r'\s+', ' ', normalized)

        return normalized

    def generate_llm_first_annotations(self, instruction: str, objects: List[Dict[str, Any]]) -> List[str]:
        """LLM-First 파이프라인용 주석 생성"""
        action = self.extract_action_intent(instruction)
        extracted_objects = self.extract_objects(instruction)

        annotations = [
            f"원본 명령: {instruction}",
            f"동작 의도: {action}",
            f"대상 객체: {', '.join(extracted_objects)}",
            f"정규화된 명령: {self.normalize_instruction(instruction)}"
        ]

        # 물리 속성 기반 주석 추가
        if objects:
            for obj in objects:
                if 'material' in obj:
                    annotations.append(f"재료 속성: {obj.get('name', 'unknown')} - {obj['material']}")
                if 'mass' in obj:
                    annotations.append(f"질량 정보: {obj.get('name', 'unknown')} - {obj['mass']}kg")

        return annotations

class TrajectoryProcessor:
    """궤적 처리기"""

    def __init__(self, target_frequency: float = 100.0):
        """프로세서 초기화"""
        self.target_frequency = target_frequency  # Hz
        self.target_dt = 1.0 / target_frequency

    def resample_trajectory(self, trajectory: np.ndarray, timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """궤적 리샘플링"""
        if len(timestamps) < 2:
            return trajectory, timestamps

        # 목표 타임스탬프 생성
        total_time = timestamps[-1] - timestamps[0]
        num_points = int(total_time * self.target_frequency) + 1
        new_timestamps = np.linspace(timestamps[0], timestamps[-1], num_points)

        # 각 차원별로 보간
        new_trajectory = np.zeros((num_points, trajectory.shape[1]))

        for i in range(trajectory.shape[1]):
            f = interp1d(timestamps, trajectory[:, i], kind='cubic',
                        bounds_error=False, fill_value='extrapolate')
            new_trajectory[:, i] = f(new_timestamps)

        return new_trajectory, new_timestamps

    def smooth_trajectory(self, trajectory: np.ndarray, window_size: int = 5) -> np.ndarray:
        """궤적 평활화 (이동 평균)"""
        if len(trajectory) < window_size:
            return trajectory

        smoothed = trajectory.copy()
        half_window = window_size // 2

        for i in range(trajectory.shape[1]):
            # 패딩 적용
            padded = np.pad(trajectory[:, i], (half_window, half_window), mode='edge')

            # 이동 평균 계산
            for j in range(len(trajectory)):
                start_idx = j
                end_idx = j + window_size
                smoothed[j, i] = np.mean(padded[start_idx:end_idx])

        return smoothed

    def compute_velocity(self, trajectory: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """속도 계산"""
        if len(trajectory) < 2:
            return np.zeros_like(trajectory)

        dt = np.diff(timestamps)
        dt = np.append(dt, dt[-1])  # 마지막 점을 위한 확장

        velocity = np.zeros_like(trajectory)
        velocity[1:] = np.diff(trajectory, axis=0) / dt[1:, np.newaxis]
        velocity[0] = velocity[1]  # 첫 번째 점

        return velocity

class GenesisExporter:
    """Genesis AI 형식 출력기"""

    def __init__(self):
        """출력기 초기화"""
        pass

    def create_robot_config(self, robot_type: str) -> Dict[str, Any]:
        """로봇 설정 생성"""
        if robot_type == "franka_panda":
            return {
                "robot_type": "franka_panda",
                "dof": 7,
                "joint_names": [f"panda_joint{i+1}" for i in range(7)],
                "end_effector": "panda_hand",
                "base_link": "panda_link0",
                "urdf_path": "franka_panda/panda.urdf",
                "control_mode": "position",
                "max_velocity": [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100],
                "max_acceleration": [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]
            }
        else:
            return {
                "robot_type": robot_type,
                "dof": 7,  # 기본값
                "joint_names": [f"joint_{i+1}" for i in range(7)],
                "end_effector": "gripper",
                "conversion_required": True
            }

    def create_scene_description(self, scene_info: Dict[str, Any], objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """씬 설명 생성"""
        return {
            "environment_type": scene_info.get("environment", "tabletop"),
            "lighting": {
                "type": "default",
                "intensity": 1.0,
                "direction": [0, 0, -1]
            },
            "camera": {
                "position": [1.0, 1.0, 1.0],
                "target": [0, 0, 0.5],
                "fov": 45
            },
            "objects": objects,
            "workspace": {
                "center": [0.5, 0.0, 0.3],
                "size": [1.0, 1.0, 0.6]
            },
            "physics": {
                "gravity": [0, 0, -9.81],
                "time_step": 0.01,
                "solver_iterations": 10
            }
        }

    def export_episode(self, genesis_episode: GenesisEpisode, output_path: str) -> bool:
        """에피소드를 Genesis AI 형식으로 출력"""
        try:
            export_data = {
                "episode_metadata": {
                    "id": genesis_episode.episode_id,
                    "format_version": "1.0.0",
                    "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "source": "DROID_to_Genesis_converter"
                },
                "robot_configuration": genesis_episode.robot_config,
                "trajectory_data": {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in genesis_episode.trajectory.items()
                },
                "physics_properties": genesis_episode.physics_properties,
                "language_annotations": genesis_episode.language_annotations,
                "scene_description": genesis_episode.scene_description,
                "conversion_metadata": genesis_episode.metadata
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            return True

        except Exception as e:
            logger.error(f"Export failed for episode {genesis_episode.episode_id}: {e}")
            return False

class DroidToGenesisConverter:
    """DROID → Genesis AI 변환 파이프라인"""

    def __init__(self, output_dir: str = "/root/gen/converted_episodes"):
        """변환기 초기화"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 구성 요소 초기화
        self.coordinate_transformer = CoordinateTransformer()
        self.franka_mapper = FrankaKinematicMapper()
        self.physics_mapper = PhysicsPropertyMapper()
        self.language_processor = LanguageProcessor()
        self.trajectory_processor = TrajectoryProcessor()
        self.genesis_exporter = GenesisExporter()

        # 변환 메트릭
        self.metrics = ConversionMetrics()

        logger.info(f"DROID → Genesis AI 변환기 초기화 완료")
        logger.info(f"출력 디렉토리: {self.output_dir}")

    def convert_episode(self, droid_episode: DroidEpisode) -> Optional[GenesisEpisode]:
        """단일 에피소드 변환"""
        start_time = time.time()

        try:
            logger.info(f"에피소드 변환 시작: {droid_episode.episode_id}")

            # 1. 좌표계 변환
            transformed_joint_positions = droid_episode.joint_positions
            transformed_ee_poses = self.coordinate_transformer.transform_trajectory(
                droid_episode.end_effector_poses
            )

            # 2. Franka 키네마틱 검증 및 수정
            if droid_episode.robot_type == "franka_panda":
                is_valid, violations = self.franka_mapper.validate_joint_limits(transformed_joint_positions)
                if not is_valid:
                    logger.warning(f"관절 한계 위반 감지: {violations}")
                    transformed_joint_positions = self.franka_mapper.clamp_joint_positions(transformed_joint_positions)

            # 3. 궤적 처리
            resampled_joints, new_timestamps = self.trajectory_processor.resample_trajectory(
                transformed_joint_positions, droid_episode.timestamps
            )
            resampled_ee_poses, _ = self.trajectory_processor.resample_trajectory(
                transformed_ee_poses, droid_episode.timestamps
            )

            smoothed_joints = self.trajectory_processor.smooth_trajectory(resampled_joints)
            smoothed_ee_poses = self.trajectory_processor.smooth_trajectory(resampled_ee_poses)

            # 4. 속도 계산
            joint_velocities = self.trajectory_processor.compute_velocity(smoothed_joints, new_timestamps)
            ee_velocities = self.trajectory_processor.compute_velocity(smoothed_ee_poses, new_timestamps)

            # 5. 물리 속성 매핑
            genesis_objects = []
            for obj in droid_episode.objects:
                genesis_obj = self.physics_mapper.map_object_properties(obj)
                genesis_obj['name'] = obj.get('name', 'unknown_object')
                genesis_objects.append(genesis_obj)

            # 6. 언어 처리
            language_annotations = self.language_processor.generate_llm_first_annotations(
                droid_episode.language_instruction, droid_episode.objects
            )

            # 7. Genesis AI 에피소드 구성
            genesis_episode = GenesisEpisode(
                episode_id=f"genesis_{droid_episode.episode_id}",
                robot_config=self.genesis_exporter.create_robot_config(droid_episode.robot_type),
                trajectory={
                    "joint_positions": smoothed_joints,
                    "joint_velocities": joint_velocities,
                    "end_effector_poses": smoothed_ee_poses,
                    "end_effector_velocities": ee_velocities,
                    "gripper_states": droid_episode.gripper_states,
                    "timestamps": new_timestamps
                },
                physics_properties={
                    "objects": genesis_objects,
                    "environment": droid_episode.scene_info
                },
                language_annotations=language_annotations,
                scene_description=self.genesis_exporter.create_scene_description(
                    droid_episode.scene_info, genesis_objects
                ),
                metadata={
                    "source_episode_id": droid_episode.episode_id,
                    "source_robot_type": droid_episode.robot_type,
                    "conversion_time": time.time() - start_time,
                    "original_trajectory_length": len(droid_episode.joint_positions),
                    "converted_trajectory_length": len(smoothed_joints),
                    "coordinate_transform_applied": True,
                    "trajectory_resampled": True,
                    "trajectory_smoothed": True
                }
            )

            # 변환 메트릭 업데이트
            self.metrics.successful_conversions += 1
            self.metrics.avg_trajectory_length += len(smoothed_joints)
            self.metrics.avg_processing_time += time.time() - start_time

            logger.info(f"에피소드 변환 완료: {droid_episode.episode_id} -> {genesis_episode.episode_id}")
            return genesis_episode

        except Exception as e:
            logger.error(f"에피소드 변환 실패: {droid_episode.episode_id}, 오류: {e}")
            self.metrics.failed_conversions += 1
            return None

    def convert_batch(self, droid_episodes: List[DroidEpisode]) -> List[GenesisEpisode]:
        """배치 변환"""
        logger.info(f"배치 변환 시작: {len(droid_episodes)}개 에피소드")

        self.metrics.total_episodes = len(droid_episodes)
        converted_episodes = []

        for i, episode in enumerate(droid_episodes):
            logger.info(f"진행률: {i+1}/{len(droid_episodes)}")

            genesis_episode = self.convert_episode(episode)
            if genesis_episode:
                converted_episodes.append(genesis_episode)

                # 에피소드별 파일로 저장
                output_path = self.output_dir / f"{genesis_episode.episode_id}.json"
                self.genesis_exporter.export_episode(genesis_episode, str(output_path))

        # 평균 메트릭 계산
        if self.metrics.successful_conversions > 0:
            self.metrics.avg_trajectory_length /= self.metrics.successful_conversions
            self.metrics.avg_processing_time /= self.metrics.successful_conversions

        logger.info(f"배치 변환 완료: {len(converted_episodes)}개 성공, {self.metrics.failed_conversions}개 실패")
        return converted_episodes

    def export_conversion_report(self, output_path: str = "/root/gen/conversion_report.json"):
        """변환 보고서 생성"""
        report = {
            "conversion_summary": {
                "total_episodes": self.metrics.total_episodes,
                "successful_conversions": self.metrics.successful_conversions,
                "failed_conversions": self.metrics.failed_conversions,
                "success_rate": self.metrics.successful_conversions / max(1, self.metrics.total_episodes)
            },
            "performance_metrics": {
                "avg_trajectory_length": self.metrics.avg_trajectory_length,
                "avg_processing_time_seconds": self.metrics.avg_processing_time,
                "total_processing_time": self.metrics.avg_processing_time * self.metrics.total_episodes
            },
            "conversion_details": {
                "coordinate_transform": "DROID ROS → Genesis AI",
                "kinematic_validation": "Franka Panda joint limits",
                "trajectory_processing": "Resampling + Smoothing at 100Hz",
                "physics_mapping": "Material-based property conversion",
                "language_processing": "LLM-First annotation generation"
            },
            "output_location": str(self.output_dir),
            "report_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"변환 보고서 저장: {output_path}")
        return report

def main():
    """메인 실행 함수"""
    print("🚀 DROID → Genesis AI + Franka 변환 파이프라인 시작!")

    # 변환기 초기화
    converter = DroidToGenesisConverter()

    # 시뮬레이션 DROID 데이터 생성 (실제 구현시 DROID 로더 사용)
    print("📊 시뮬레이션 DROID 데이터 생성 중...")

    droid_episodes = []
    for i in range(3):
        # 시뮬레이션 궤적 생성
        trajectory_length = np.random.randint(50, 150)
        joint_positions = np.random.uniform(-1, 1, (trajectory_length, 7))
        ee_poses = np.random.uniform(-1, 1, (trajectory_length, 7))
        gripper_states = np.random.choice([0, 1], trajectory_length)
        timestamps = np.linspace(0, trajectory_length * 0.01, trajectory_length)

        episode = DroidEpisode(
            episode_id=f"droid_episode_{i:03d}",
            robot_type="franka_panda",
            language_instruction=f"Pick up the object and place it in the container {i}",
            joint_positions=joint_positions,
            end_effector_poses=ee_poses,
            gripper_states=gripper_states,
            timestamps=timestamps,
            objects=[
                {"name": f"object_{i}", "material": "plastic", "mass": 0.1},
                {"name": f"container_{i}", "material": "metal", "mass": 0.5}
            ],
            scene_info={"environment": "tabletop", "lighting": "normal"}
        )
        droid_episodes.append(episode)

    print(f"✓ {len(droid_episodes)}개 시뮬레이션 에피소드 생성 완료")

    # 배치 변환 실행
    converted_episodes = converter.convert_batch(droid_episodes)

    # 변환 보고서 생성
    report = converter.export_conversion_report()

    # 결과 출력
    print("\n" + "="*60)
    print("🎯 DROID → Genesis AI 변환 결과")
    print("="*60)
    print(f"총 에피소드: {report['conversion_summary']['total_episodes']}")
    print(f"성공한 변환: {report['conversion_summary']['successful_conversions']}")
    print(f"실패한 변환: {report['conversion_summary']['failed_conversions']}")
    print(f"성공률: {report['conversion_summary']['success_rate']:.1%}")
    print(f"평균 처리 시간: {report['performance_metrics']['avg_processing_time_seconds']:.3f}초")
    print(f"출력 위치: {report['output_location']}")

    print(f"\n✅ 변환 파이프라인 실행 완료!")

if __name__ == "__main__":
    main()