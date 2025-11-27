#!/usr/bin/env python3
"""
파이프라인 테스트용 샘플 데이터 생성기

PhysicalAI 형태의 가상 데이터셋을 생성하여
파이프라인 테스트에 사용합니다.
"""

import json
import h5py
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SampleDataGenerator:
    """샘플 데이터 생성기"""

    def __init__(self, output_dir: str = "data/test_samples"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_franka_episode_data(self, episode_id: int = 1) -> dict:
        """Franka 로봇 에피소드 데이터 생성"""

        # 시뮬레이션 파라미터
        duration = 5.0  # 5초
        dt = 0.01  # 10ms
        num_steps = int(duration / dt)

        # 시간 배열
        timestamps = np.linspace(0, duration, num_steps)

        # 7-DOF Franka 관절 데이터 생성
        joint_count = 7

        # 간단한 움직임 패턴 (사인파 기반)
        joint_positions = np.zeros((num_steps, joint_count))
        joint_velocities = np.zeros((num_steps, joint_count))
        joint_torques = np.zeros((num_steps, joint_count))

        for i in range(joint_count):
            # 각 관절마다 다른 주파수와 진폭
            freq = 0.5 + i * 0.1  # Hz
            amp = 0.3 + i * 0.05  # radians

            joint_positions[:, i] = amp * np.sin(2 * np.pi * freq * timestamps)
            joint_velocities[:, i] = amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq * timestamps)
            joint_torques[:, i] = np.random.normal(0, 0.1, num_steps)  # 노이즈 추가

        # 말단장치 위치 (간단한 추정)
        end_effector_poses = np.zeros((num_steps, 4, 4))  # 동차좌표 변환행렬
        for t in range(num_steps):
            # 간단한 말단장치 위치 계산 (실제로는 순기구학 필요)
            x = 0.5 + 0.1 * np.sin(timestamps[t])
            y = 0.0 + 0.1 * np.cos(timestamps[t])
            z = 0.3 + 0.05 * np.sin(2 * timestamps[t])

            # 4x4 변환 행렬
            T = np.eye(4)
            T[0:3, 3] = [x, y, z]
            end_effector_poses[t] = T

        return {
            'episode_id': episode_id,
            'timestamps': timestamps,
            'joint_positions': joint_positions,
            'joint_velocities': joint_velocities,
            'joint_torques': joint_torques,
            'end_effector_poses': end_effector_poses,
            'sampling_rate': 1.0 / dt
        }

    def generate_physics_properties(self, scenario: str = "manipulation") -> dict:
        """물리 속성 데이터 생성"""

        scenarios = {
            "manipulation": {
                "mass": 1.2,  # kg
                "friction": 0.7,
                "restitution": 0.3,
                "linear_damping": 0.1,
                "angular_damping": 0.1,
                "material_type": "plastic",
                "surface_roughness": 0.001,  # m
                "density": 950.0  # kg/m³
            },
            "assembly": {
                "mass": 0.8,
                "friction": 0.9,
                "restitution": 0.1,
                "linear_damping": 0.05,
                "angular_damping": 0.05,
                "material_type": "metal",
                "surface_roughness": 0.0005,
                "density": 2700.0
            },
            "grasping": {
                "mass": 0.3,
                "friction": 0.8,
                "restitution": 0.4,
                "linear_damping": 0.2,
                "angular_damping": 0.2,
                "material_type": "rubber",
                "surface_roughness": 0.002,
                "density": 1200.0
            }
        }

        base_props = scenarios.get(scenario, scenarios["manipulation"])

        # 약간의 노이즈 추가
        props = base_props.copy()
        props["mass"] += np.random.normal(0, 0.05)
        props["friction"] += np.random.normal(0, 0.02)
        props["mass"] = max(0.1, props["mass"])  # 최소값 보장
        props["friction"] = np.clip(props["friction"], 0.1, 1.5)  # 범위 제한

        return props

    def generate_scene_description(self, task_type: str = "pick_and_place") -> dict:
        """장면 설명 데이터 생성"""

        task_descriptions = {
            "pick_and_place": {
                "task_description": "Pick up the object from the table and place it in the target location",
                "environment_type": "laboratory",
                "lighting_conditions": "normal",
                "objects": [
                    {"name": "target_object", "type": "box", "size": [0.05, 0.05, 0.1]},
                    {"name": "table", "type": "plane", "size": [1.0, 1.0, 0.02]}
                ],
                "obstacles": [],
                "success_criteria": ["object_at_target", "stable_grasp", "no_collisions"]
            },
            "assembly": {
                "task_description": "Assemble the components into the target configuration",
                "environment_type": "workshop",
                "lighting_conditions": "bright",
                "objects": [
                    {"name": "part_a", "type": "cylinder", "size": [0.02, 0.1]},
                    {"name": "part_b", "type": "box", "size": [0.08, 0.08, 0.03]}
                ],
                "obstacles": [
                    {"name": "fixture", "type": "box", "size": [0.1, 0.1, 0.05]}
                ],
                "success_criteria": ["parts_assembled", "proper_alignment", "secure_connection"]
            },
            "sorting": {
                "task_description": "Sort objects by color and size into designated containers",
                "environment_type": "warehouse",
                "lighting_conditions": "normal",
                "objects": [
                    {"name": "red_box", "type": "box", "size": [0.03, 0.03, 0.03]},
                    {"name": "blue_cylinder", "type": "cylinder", "size": [0.025, 0.05]},
                    {"name": "container_a", "type": "box", "size": [0.15, 0.15, 0.08]},
                    {"name": "container_b", "type": "box", "size": [0.15, 0.15, 0.08]}
                ],
                "obstacles": [],
                "success_criteria": ["correct_sorting", "all_objects_placed", "containers_stable"]
            }
        }

        return task_descriptions.get(task_type, task_descriptions["pick_and_place"])

    def generate_json_dataset(self, num_episodes: int = 5, filename: str = None) -> str:
        """JSON 형태의 데이터셋 생성"""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"physicalai_sample_dataset_{timestamp}.json"

        dataset = {
            "metadata": {
                "dataset_name": "PhysicalAI Sample Dataset",
                "creation_time": datetime.now().isoformat(),
                "num_episodes": num_episodes,
                "robot_type": "franka_panda",
                "simulation_engine": "isaac_sim",
                "version": "1.0.0"
            },
            "episodes": []
        }

        # 시나리오와 작업 타입 목록
        scenarios = ["manipulation", "assembly", "grasping"]
        task_types = ["pick_and_place", "assembly", "sorting"]

        for i in range(num_episodes):
            logger.info(f"에피소드 {i+1}/{num_episodes} 생성 중...")

            # 랜덤하게 시나리오와 작업 선택
            scenario = np.random.choice(scenarios)
            task_type = np.random.choice(task_types)

            # 로봇 데이터 생성
            robot_data = self.generate_franka_episode_data(i+1)

            # 물리 속성 생성
            physics_props = self.generate_physics_properties(scenario)

            # 장면 설명 생성
            scene_desc = self.generate_scene_description(task_type)

            episode = {
                "episode_id": i + 1,
                "scenario": scenario,
                "robot": {
                    "type": "franka_panda",
                    "dof": 7,
                    "timestamps": robot_data["timestamps"].tolist(),
                    "joint_positions": robot_data["joint_positions"].tolist(),
                    "joint_velocities": robot_data["joint_velocities"].tolist(),
                    "joint_torques": robot_data["joint_torques"].tolist(),
                    "end_effector_poses": robot_data["end_effector_poses"].tolist(),
                    "sampling_rate": robot_data["sampling_rate"]
                },
                "physics": physics_props,
                "scene": scene_desc,
                "metadata": {
                    "success": bool(np.random.choice([True, False], p=[0.8, 0.2])),
                    "execution_time": float(robot_data["timestamps"][-1]),
                    "notes": f"Generated episode for {scenario} scenario with {task_type} task"
                }
            }

            dataset["episodes"].append(episode)

        # JSON 파일 저장
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)

        logger.info(f"JSON 데이터셋 저장완료: {output_path}")
        logger.info(f"크기: {output_path.stat().st_size / 1024:.1f} KB")

        return str(output_path)

    def generate_hdf5_dataset(self, num_episodes: int = 3, filename: str = None) -> str:
        """HDF5 형태의 데이터셋 생성"""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"physicalai_sample_dataset_{timestamp}.h5"

        output_path = self.output_dir / filename

        with h5py.File(output_path, 'w') as f:
            # 메타데이터 그룹
            meta_group = f.create_group("metadata")
            meta_group.attrs["dataset_name"] = "PhysicalAI Sample HDF5 Dataset"
            meta_group.attrs["creation_time"] = datetime.now().isoformat()
            meta_group.attrs["num_episodes"] = num_episodes
            meta_group.attrs["robot_type"] = "franka_panda"

            for i in range(num_episodes):
                logger.info(f"HDF5 에피소드 {i+1}/{num_episodes} 생성 중...")

                # 에피소드 그룹 생성
                episode_group = f.create_group(f"episode_{i+1:03d}")

                # 로봇 데이터
                robot_data = self.generate_franka_episode_data(i+1)
                robot_group = episode_group.create_group("robot")

                robot_group.create_dataset("timestamps", data=robot_data["timestamps"])
                robot_group.create_dataset("joint_positions", data=robot_data["joint_positions"])
                robot_group.create_dataset("joint_velocities", data=robot_data["joint_velocities"])
                robot_group.create_dataset("joint_torques", data=robot_data["joint_torques"])
                robot_group.create_dataset("end_effector_poses", data=robot_data["end_effector_poses"])
                robot_group.attrs["sampling_rate"] = robot_data["sampling_rate"]

                # 물리 속성
                physics_props = self.generate_physics_properties()
                physics_group = episode_group.create_group("physics")
                for key, value in physics_props.items():
                    if isinstance(value, str):
                        physics_group.attrs[key] = value
                    else:
                        physics_group.attrs[key] = value

                # 장면 정보
                scene_desc = self.generate_scene_description()
                scene_group = episode_group.create_group("scene")
                scene_group.attrs["task_description"] = scene_desc["task_description"]
                scene_group.attrs["environment_type"] = scene_desc["environment_type"]

                # 객체 정보 (JSON 문자열로 저장)
                scene_group.attrs["objects"] = json.dumps(scene_desc["objects"])
                scene_group.attrs["success_criteria"] = json.dumps(scene_desc["success_criteria"])

        logger.info(f"HDF5 데이터셋 저장완료: {output_path}")
        logger.info(f"크기: {output_path.stat().st_size / 1024:.1f} KB")

        return str(output_path)

    def validate_generated_data(self, file_path: str) -> bool:
        """생성된 데이터 검증"""

        try:
            path = Path(file_path)

            if path.suffix == '.json':
                return self._validate_json_data(path)
            elif path.suffix in ['.h5', '.hdf5']:
                return self._validate_hdf5_data(path)
            else:
                logger.error(f"지원하지 않는 파일 형식: {path.suffix}")
                return False

        except Exception as e:
            logger.error(f"데이터 검증 실패: {e}")
            return False

    def _validate_json_data(self, path: Path) -> bool:
        """JSON 데이터 검증"""

        with open(path, 'r') as f:
            data = json.load(f)

        # 기본 구조 확인
        required_keys = ["metadata", "episodes"]
        for key in required_keys:
            if key not in data:
                logger.error(f"필수 키 누락: {key}")
                return False

        # 에피소드 데이터 확인
        episodes = data["episodes"]
        if not episodes:
            logger.error("에피소드 데이터가 없습니다")
            return False

        # 첫 번째 에피소드 상세 검증
        episode = episodes[0]
        required_episode_keys = ["robot", "physics", "scene"]

        for key in required_episode_keys:
            if key not in episode:
                logger.error(f"에피소드에서 필수 키 누락: {key}")
                return False

        # 로봇 데이터 검증
        robot = episode["robot"]
        joint_positions = np.array(robot["joint_positions"])
        timestamps = np.array(robot["timestamps"])

        if joint_positions.shape[0] != len(timestamps):
            logger.error("관절 위치와 시간 데이터 길이 불일치")
            return False

        if joint_positions.shape[1] != 7:  # Franka = 7 DOF
            logger.error(f"관절 수 불일치: 예상 7, 실제 {joint_positions.shape[1]}")
            return False

        logger.info("JSON 데이터 검증 성공")
        return True

    def _validate_hdf5_data(self, path: Path) -> bool:
        """HDF5 데이터 검증"""

        with h5py.File(path, 'r') as f:
            # 메타데이터 확인
            if "metadata" not in f:
                logger.error("메타데이터 그룹 누락")
                return False

            # 에피소드 그룹 확인
            episode_groups = [key for key in f.keys() if key.startswith("episode_")]
            if not episode_groups:
                logger.error("에피소드 그룹이 없습니다")
                return False

            # 첫 번째 에피소드 검증
            episode_group = f[episode_groups[0]]

            required_groups = ["robot", "physics", "scene"]
            for group_name in required_groups:
                if group_name not in episode_group:
                    logger.error(f"에피소드에서 그룹 누락: {group_name}")
                    return False

            # 로봇 데이터 검증
            robot_group = episode_group["robot"]

            required_datasets = ["timestamps", "joint_positions", "joint_velocities"]
            for dataset_name in required_datasets:
                if dataset_name not in robot_group:
                    logger.error(f"로봇 그룹에서 데이터셋 누락: {dataset_name}")
                    return False

            # 데이터 크기 확인
            timestamps = robot_group["timestamps"][:]
            joint_positions = robot_group["joint_positions"][:]

            if len(timestamps) != joint_positions.shape[0]:
                logger.error("HDF5: 시간과 관절 위치 데이터 길이 불일치")
                return False

        logger.info("HDF5 데이터 검증 성공")
        return True


def main():
    """메인 실행 함수"""
    logger.info("샘플 데이터 생성 시작")

    generator = SampleDataGenerator()

    # JSON 데이터셋 생성
    logger.info("\n=== JSON 데이터셋 생성 ===")
    json_path = generator.generate_json_dataset(num_episodes=3)
    json_valid = generator.validate_generated_data(json_path)

    # HDF5 데이터셋 생성
    logger.info("\n=== HDF5 데이터셋 생성 ===")
    hdf5_path = generator.generate_hdf5_dataset(num_episodes=2)
    hdf5_valid = generator.validate_generated_data(hdf5_path)

    # 결과 요약
    logger.info("\n" + "="*50)
    logger.info("샘플 데이터 생성 결과:")
    logger.info(f"  JSON 데이터셋: {'✓ 성공' if json_valid else '✗ 실패'} ({json_path})")
    logger.info(f"  HDF5 데이터셋: {'✓ 성공' if hdf5_valid else '✗ 실패'} ({hdf5_path})")

    if json_valid and hdf5_valid:
        logger.info("\n🎉 모든 샘플 데이터 생성 및 검증 완료!")
    else:
        logger.warning("\n⚠️  일부 데이터 생성에 문제가 있습니다.")

    return json_valid and hdf5_valid


if __name__ == "__main__":
    success = main()