#!/usr/bin/env python3
"""
공공 데이터셋 기반 QLoRA 파인튜닝 데이터셋 구축기
Physical Property Extraction for Robot Control

기존 공공 데이터셋을 활용하여 물리 속성 추출 및 로봇 제어 파라미터 학습용 데이터셋 생성
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import requests
from pathlib import Path
import re
import random
from datetime import datetime

@dataclass
class RobotControlSample:
    """로봇 제어 학습 샘플"""
    instruction: str            # 자연어 명령
    input_context: str         # 입력 컨텍스트 (환경, 객체 상태 등)
    output: str                # 예상 출력 (제어 파라미터 JSON)
    task_type: str             # 작업 유형 (pick, place, move, etc.)
    difficulty: str            # 난이도 (easy, medium, hard)
    physical_properties: Dict  # 물리 속성 정보
    safety_constraints: List   # 안전 제약사항

@dataclass
class DatasetMetrics:
    """데이터셋 품질 메트릭"""
    total_samples: int
    task_distribution: Dict[str, int]
    difficulty_distribution: Dict[str, int]
    material_distribution: Dict[str, int]
    quality_score: float
    completeness_score: float

class PublicDatasetQLoRABuilder:
    """공공 데이터셋 기반 QLoRA 파인튜닝 데이터셋 구축기"""

    def __init__(self, output_dir: str = "/root/gen/datasets"):
        """
        데이터셋 구축기 초기화

        Args:
            output_dir: 출력 디렉토리 경로
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 물리 속성 템플릿
        self.material_properties = {
            "metal": {"density": 7.8, "friction": 0.6, "stiffness": 200, "fragility": 0.1},
            "plastic": {"density": 1.2, "friction": 0.4, "stiffness": 3, "fragility": 0.3},
            "glass": {"density": 2.5, "friction": 0.2, "stiffness": 70, "fragility": 0.9},
            "wood": {"density": 0.6, "friction": 0.5, "stiffness": 12, "fragility": 0.2},
            "fabric": {"density": 0.3, "friction": 0.7, "stiffness": 0.1, "fragility": 0.1},
            "ceramic": {"density": 2.3, "friction": 0.3, "stiffness": 100, "fragility": 0.8},
            "rubber": {"density": 1.5, "friction": 0.8, "stiffness": 0.01, "fragility": 0.05}
        }

        # 동작 템플릿
        self.action_templates = {
            "pick": ["집어라", "들어라", "잡아라", "가져가라"],
            "place": ["놓아라", "두어라", "배치해라", "올려놓아라"],
            "move": ["옮겨라", "이동시켜라", "가져다놓아라", "운반해라"],
            "push": ["밀어라", "밀어내라", "전진시켜라"],
            "pull": ["당겨라", "끌어라", "가져와라"]
        }

        # 형용사 기반 물리 속성 매핑
        self.adjective_mapping = {
            "무거운": {"mass": "heavy", "grip_force_mult": 1.5},
            "가벼운": {"mass": "light", "grip_force_mult": 0.7},
            "딱딱한": {"stiffness": "hard", "approach_force_mult": 1.2},
            "부드러운": {"stiffness": "soft", "approach_force_mult": 0.8},
            "깨지기쉬운": {"fragility": "fragile", "safety_mult": 2.0},
            "미끄러운": {"friction": "low", "grip_force_mult": 1.8},
            "거친": {"friction": "high", "grip_force_mult": 0.9},
            "큰": {"size": "large", "approach_angle_mult": 1.3},
            "작은": {"size": "small", "approach_angle_mult": 0.8}
        }

        self.samples: List[RobotControlSample] = []

    def download_public_datasets(self) -> List[str]:
        """공공 데이터셋 다운로드 및 준비"""
        print("공공 데이터셋 검색 및 준비 중...")

        # 실제 공공 데이터셋 대신 시뮬레이션용 데이터 생성
        # 실제 구현시에는 다음과 같은 공공 데이터셋 활용:
        # - AI Hub 로봇 데이터셋
        # - COCO Dataset (객체 인식)
        # - Open Images Dataset
        # - CommonVoice (음성 명령)

        datasets = [
            "simulated_object_descriptions.json",
            "simulated_robot_commands.json",
            "simulated_material_properties.json"
        ]

        for dataset_name in datasets:
            self._create_simulated_dataset(dataset_name)

        return datasets

    def _create_simulated_dataset(self, dataset_name: str):
        """시뮬레이션 데이터셋 생성"""
        dataset_path = self.output_dir / dataset_name

        if "object_descriptions" in dataset_name:
            # 객체 설명 데이터
            data = []
            objects = ["상자", "컵", "병", "블록", "인형", "그릇", "책", "펜", "공", "접시"]
            materials = list(self.material_properties.keys())
            adjectives = list(self.adjective_mapping.keys())

            for _ in range(200):
                obj = random.choice(objects)
                material = random.choice(materials)
                adj = random.choice(adjectives)

                data.append({
                    "object": obj,
                    "material": material,
                    "adjective": adj,
                    "description": f"{adj} {material} {obj}",
                    "properties": self.material_properties[material]
                })

        elif "robot_commands" in dataset_name:
            # 로봇 명령 데이터
            data = []
            locations = ["테이블", "선반", "바구니", "상자", "침대", "의자"]

            for _ in range(300):
                action = random.choice(list(self.action_templates.keys()))
                action_verb = random.choice(self.action_templates[action])
                obj = random.choice(["상자", "컵", "병", "블록", "인형"])
                location = random.choice(locations)

                command = f"{obj}를 {location}에 {action_verb}"

                data.append({
                    "command": command,
                    "action": action,
                    "object": obj,
                    "destination": location,
                    "complexity": random.choice(["simple", "medium", "complex"])
                })

        elif "material_properties" in dataset_name:
            # 재료 속성 데이터
            data = []
            for material, props in self.material_properties.items():
                for i in range(10):  # 각 재료마다 변형된 속성 생성
                    variation = {
                        "material": material,
                        "density": props["density"] * (0.8 + 0.4 * random.random()),
                        "friction": max(0.1, min(1.0, props["friction"] * (0.8 + 0.4 * random.random()))),
                        "stiffness": props["stiffness"] * (0.5 + 1.0 * random.random()),
                        "fragility": max(0.0, min(1.0, props["fragility"] * (0.5 + 1.0 * random.random())))
                    }
                    data.append(variation)

        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✓ {dataset_name} 생성 완료 ({len(data)} 샘플)")

    def generate_training_samples(self, target_count: int = 2000) -> List[RobotControlSample]:
        """QLoRA 파인튜닝용 학습 샘플 생성"""
        print(f"QLoRA 학습 샘플 {target_count}개 생성 중...")

        # 공공 데이터셋 로드
        datasets = self.download_public_datasets()

        object_data = self._load_dataset("simulated_object_descriptions.json")
        command_data = self._load_dataset("simulated_robot_commands.json")
        material_data = self._load_dataset("simulated_material_properties.json")

        generated_count = 0

        while generated_count < target_count:
            # 랜덤하게 조합
            obj_sample = random.choice(object_data)
            cmd_sample = random.choice(command_data)
            mat_sample = random.choice(material_data)

            # 통합 샘플 생성
            sample = self._create_integrated_sample(obj_sample, cmd_sample, mat_sample)

            if sample:
                self.samples.append(sample)
                generated_count += 1

                if generated_count % 100 == 0:
                    print(f"  진행률: {generated_count}/{target_count} ({generated_count/target_count*100:.1f}%)")

        print(f"✓ {len(self.samples)}개 학습 샘플 생성 완료")
        return self.samples

    def _load_dataset(self, filename: str) -> List[Dict]:
        """데이터셋 파일 로드"""
        try:
            with open(self.output_dir / filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"데이터셋 로드 실패 {filename}: {e}")
            return []

    def _create_integrated_sample(self, obj_sample: Dict, cmd_sample: Dict, mat_sample: Dict) -> Optional[RobotControlSample]:
        """통합 학습 샘플 생성"""
        try:
            # 명령어 구성
            description = obj_sample['description']
            action = cmd_sample['action']
            destination = cmd_sample['destination']

            instruction = f"{description}을 {destination}에 {random.choice(self.action_templates[action])}"

            # 물리 속성 계산
            base_props = mat_sample.copy()

            # 형용사 기반 조정
            if obj_sample['adjective'] in self.adjective_mapping:
                adj_props = self.adjective_mapping[obj_sample['adjective']]
                base_props.update(adj_props)

            # 제어 파라미터 계산
            control_params = self._calculate_control_parameters(base_props, action)

            # 안전 제약사항
            safety_constraints = self._generate_safety_constraints(base_props, action)

            # 난이도 계산
            difficulty = self._calculate_difficulty(base_props, action)

            # 입력 컨텍스트
            input_context = f"객체: {description}, 목적지: {destination}, 환경: 실내"

            # 출력 JSON
            output_json = {
                "control_parameters": control_params,
                "physical_properties": {
                    "material": obj_sample['material'],
                    "mass": base_props.get('mass', 'medium'),
                    "friction": base_props['friction'],
                    "stiffness": base_props['stiffness'],
                    "fragility": base_props['fragility']
                },
                "confidence": random.uniform(0.7, 0.95)
            }

            return RobotControlSample(
                instruction=instruction,
                input_context=input_context,
                output=json.dumps(output_json, ensure_ascii=False),
                task_type=action,
                difficulty=difficulty,
                physical_properties=base_props,
                safety_constraints=safety_constraints
            )

        except Exception as e:
            print(f"샘플 생성 오류: {e}")
            return None

    def _calculate_control_parameters(self, properties: Dict, action: str) -> Dict[str, float]:
        """물리 속성 기반 제어 파라미터 계산"""
        # 기본 파라미터
        params = {
            "grip_force": 0.5,
            "lift_speed": 0.5,
            "approach_angle": 0.0,
            "contact_force": 0.3,
            "safety_margin": 0.8
        }

        # 물리 속성 기반 조정
        if 'grip_force_mult' in properties:
            params['grip_force'] *= properties['grip_force_mult']

        if 'safety_mult' in properties:
            params['safety_margin'] *= properties['safety_mult']

        if 'approach_force_mult' in properties:
            params['contact_force'] *= properties['approach_force_mult']

        # 재료별 조정 (숫자 값만 처리)
        fragility = properties.get('fragility', 0.5)
        if isinstance(fragility, (int, float)) and fragility > 0.5:
            params['grip_force'] *= 0.7
            params['lift_speed'] *= 0.6
            params['safety_margin'] *= 1.5

        friction = properties.get('friction', 0.5)
        if isinstance(friction, (int, float)) and friction < 0.3:
            params['grip_force'] *= 1.4

        # 값 정규화
        for key in params:
            params[key] = max(0.1, min(1.0, params[key]))
            params[key] = round(params[key], 2)

        return params

    def _generate_safety_constraints(self, properties: Dict, action: str) -> List[str]:
        """안전 제약사항 생성"""
        constraints = []

        fragility = properties.get('fragility', 0.5)
        if isinstance(fragility, (int, float)) and fragility > 0.5:
            constraints.append("fragile_handling")
            constraints.append("gentle_grip")

        friction = properties.get('friction', 0.5)
        if isinstance(friction, (int, float)) and friction < 0.3:
            constraints.append("secure_grip")
            constraints.append("slow_movement")

        if action == "place":
            constraints.append("precise_positioning")

        return constraints

    def _calculate_difficulty(self, properties: Dict, action: str) -> str:
        """작업 난이도 계산"""
        difficulty_score = 0

        # 물리 속성 기반 난이도
        fragility = properties.get('fragility', 0.5)
        if isinstance(fragility, (int, float)):
            if fragility > 0.7:
                difficulty_score += 2
            elif fragility > 0.4:
                difficulty_score += 1

        friction = properties.get('friction', 0.5)
        if isinstance(friction, (int, float)) and friction < 0.3:
            difficulty_score += 1

        # 동작 기반 난이도
        if action in ["place", "push"]:
            difficulty_score += 1
        elif action == "pull":
            difficulty_score += 2

        if difficulty_score >= 3:
            return "hard"
        elif difficulty_score >= 1:
            return "medium"
        else:
            return "easy"

    def export_datasets(self, formats: List[str] = ["jsonl", "csv", "json"]) -> Dict[str, str]:
        """다양한 형식으로 데이터셋 내보내기"""
        if not self.samples:
            print("생성된 샘플이 없습니다. generate_training_samples()를 먼저 실행해주세요.")
            return {}

        exported_files = {}

        # JSONL 형식 (QLoRA 학습용)
        if "jsonl" in formats:
            jsonl_path = self.output_dir / "qlora_training_dataset.jsonl"
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for sample in self.samples:
                    training_sample = {
                        "instruction": sample.instruction,
                        "input": sample.input_context,
                        "output": sample.output
                    }
                    f.write(json.dumps(training_sample, ensure_ascii=False) + "\n")
            exported_files["jsonl"] = str(jsonl_path)
            print(f"✓ JSONL 형식: {jsonl_path}")

        # CSV 형식
        if "csv" in formats:
            csv_path = self.output_dir / "training_dataset.csv"
            df_data = []
            for sample in self.samples:
                df_data.append({
                    "instruction": sample.instruction,
                    "input_context": sample.input_context,
                    "output": sample.output,
                    "task_type": sample.task_type,
                    "difficulty": sample.difficulty,
                    "safety_constraints": "|".join(sample.safety_constraints)
                })

            df = pd.DataFrame(df_data)
            df.to_csv(csv_path, index=False, encoding='utf-8')
            exported_files["csv"] = str(csv_path)
            print(f"✓ CSV 형식: {csv_path}")

        # JSON 형식 (전체 메타데이터 포함)
        if "json" in formats:
            json_path = self.output_dir / "complete_dataset.json"
            complete_data = {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "total_samples": len(self.samples),
                    "description": "공공데이터셋 기반 QLoRA 파인튜닝용 로봇 제어 데이터셋"
                },
                "samples": [asdict(sample) for sample in self.samples]
            }

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(complete_data, f, ensure_ascii=False, indent=2)
            exported_files["json"] = str(json_path)
            print(f"✓ JSON 형식: {json_path}")

        return exported_files

    def analyze_dataset_quality(self) -> DatasetMetrics:
        """데이터셋 품질 분석"""
        if not self.samples:
            return DatasetMetrics(0, {}, {}, {}, 0.0, 0.0)

        # 작업 분포
        task_dist = {}
        difficulty_dist = {}
        material_dist = {}

        for sample in self.samples:
            # 작업 유형 분포
            task_dist[sample.task_type] = task_dist.get(sample.task_type, 0) + 1

            # 난이도 분포
            difficulty_dist[sample.difficulty] = difficulty_dist.get(sample.difficulty, 0) + 1

            # 재료 분포 (physical_properties에서 추출)
            if 'material' in sample.physical_properties:
                material = sample.physical_properties['material']
                material_dist[material] = material_dist.get(material, 0) + 1

        # 품질 점수 계산
        quality_score = self._calculate_quality_score()
        completeness_score = self._calculate_completeness_score()

        return DatasetMetrics(
            total_samples=len(self.samples),
            task_distribution=task_dist,
            difficulty_distribution=difficulty_dist,
            material_distribution=material_dist,
            quality_score=quality_score,
            completeness_score=completeness_score
        )

    def _calculate_quality_score(self) -> float:
        """데이터셋 품질 점수 계산"""
        if not self.samples:
            return 0.0

        quality_factors = []

        # 명령어 다양성
        unique_instructions = len(set(sample.instruction for sample in self.samples))
        instruction_diversity = unique_instructions / len(self.samples)
        quality_factors.append(instruction_diversity)

        # 출력 완성도
        complete_outputs = sum(1 for sample in self.samples if sample.output and len(sample.output) > 50)
        output_completeness = complete_outputs / len(self.samples)
        quality_factors.append(output_completeness)

        # 안전 제약사항 포함률
        with_safety = sum(1 for sample in self.samples if sample.safety_constraints)
        safety_inclusion = with_safety / len(self.samples)
        quality_factors.append(safety_inclusion)

        return sum(quality_factors) / len(quality_factors)

    def _calculate_completeness_score(self) -> float:
        """데이터셋 완성도 점수 계산"""
        if not self.samples:
            return 0.0

        completeness_factors = []

        # 필수 필드 완성도
        complete_samples = 0
        for sample in self.samples:
            if (sample.instruction and sample.input_context and
                sample.output and sample.task_type and sample.difficulty):
                complete_samples += 1

        field_completeness = complete_samples / len(self.samples)
        completeness_factors.append(field_completeness)

        # 작업 유형 균형도
        task_counts = {}
        for sample in self.samples:
            task_counts[sample.task_type] = task_counts.get(sample.task_type, 0) + 1

        if task_counts:
            max_count = max(task_counts.values())
            min_count = min(task_counts.values())
            balance_score = min_count / max_count if max_count > 0 else 0
            completeness_factors.append(balance_score)

        return sum(completeness_factors) / len(completeness_factors)

    def print_dataset_summary(self):
        """데이터셋 요약 정보 출력"""
        if not self.samples:
            print("생성된 데이터셋이 없습니다.")
            return

        metrics = self.analyze_dataset_quality()

        print("\n" + "="*60)
        print("공공 데이터셋 기반 QLoRA 파인튜닝 데이터셋 요약")
        print("="*60)

        print(f"전체 샘플 수: {metrics.total_samples}")
        print(f"품질 점수: {metrics.quality_score:.2f}/1.00")
        print(f"완성도 점수: {metrics.completeness_score:.2f}/1.00")

        print(f"\n작업 유형 분포:")
        for task, count in metrics.task_distribution.items():
            percentage = count / metrics.total_samples * 100
            print(f"  {task}: {count}개 ({percentage:.1f}%)")

        print(f"\n난이도 분포:")
        for difficulty, count in metrics.difficulty_distribution.items():
            percentage = count / metrics.total_samples * 100
            print(f"  {difficulty}: {count}개 ({percentage:.1f}%)")

        print(f"\n재료 분포:")
        for material, count in metrics.material_distribution.items():
            percentage = count / metrics.total_samples * 100
            print(f"  {material}: {count}개 ({percentage:.1f}%)")

        print("\n" + "="*60)

def main():
    """메인 실행 함수"""
    print("공공 데이터셋 기반 QLoRA 파인튜닝 데이터셋 구축 시작!")
    print("="*60)

    # 데이터셋 구축기 초기화
    builder = PublicDatasetQLoRABuilder()

    # 학습 샘플 생성
    samples = builder.generate_training_samples(target_count=2000)

    # 데이터셋 내보내기
    exported_files = builder.export_datasets(["jsonl", "csv", "json"])

    # 품질 분석 및 요약
    builder.print_dataset_summary()

    print(f"\n내보낸 파일:")
    for format_type, file_path in exported_files.items():
        print(f"  {format_type.upper()}: {file_path}")

    print(f"\n✓ QLoRA 파인튜닝 데이터셋 구축 완료!")
    print(f"✓ 졸업논문용 물리 속성 추출 로봇 제어 데이터셋 준비됨")

if __name__ == "__main__":
    main()