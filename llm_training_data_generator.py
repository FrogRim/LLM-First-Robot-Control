"""
LLM Training Data Generator
DROID 에피소드 → LLM Fine-tuning 학습 데이터 생성기

목적: 물리 도메인 LLM Fine-tuning을 위한 Instruction-Response 데이터셋 생성
"""

import json
import random
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PhysicalProperties:
    """물리 속성 정의"""
    material: str
    mass_category: str
    friction: str
    fragility: str
    stiffness: str
    confidence: float


@dataclass
class ControlParameters:
    """제어 파라미터"""
    grip_force: float
    lift_speed: float
    approach_angle: float
    contact_force: float
    safety_margin: float


class CommandParaphraser:
    """자연어 명령 변형기 (데이터 증강)"""

    # Pick 동작 변형
    PICK_VARIATIONS = [
        "Pick up the {object}",
        "Grasp the {object}",
        "Lift the {object}",
        "Take the {object}",
        "Grab the {object}",
        "Hold the {object}",
        "Pick the {object}",
    ]

    # Place 동작 변형
    PLACE_VARIATIONS = [
        "place it in the {destination}",
        "put it in the {destination}",
        "move it to the {destination}",
        "set it in the {destination}",
        "position it in the {destination}",
        "drop it in the {destination}",
    ]

    # 객체 종류
    OBJECTS = [
        "object", "item", "unknown object", "target object",
        "plastic bottle", "metal can", "wooden block", "glass cup",
        "rubber ball", "ceramic mug", "fabric bag"
    ]

    # 목적지
    DESTINATIONS = [
        "container", "box", "basket", "bin", "tray", "platform", "holder"
    ]

    @classmethod
    def generate_variations(cls, base_command: str, num_variations: int = 5) -> List[str]:
        """명령어 변형 생성"""
        variations = [base_command]  # 원본 포함

        for _ in range(num_variations - 1):
            pick_template = random.choice(cls.PICK_VARIATIONS)
            place_template = random.choice(cls.PLACE_VARIATIONS)
            obj = random.choice(cls.OBJECTS)
            dest = random.choice(cls.DESTINATIONS)

            command = f"{pick_template.format(object=obj)} and {place_template.format(destination=dest)}"
            variations.append(command)

        return variations


class PhysicsVariationGenerator:
    """물리 속성 변형 생성기"""

    MATERIAL_PROPERTIES = {
        "plastic": {
            "mass_category": "light",
            "friction": "normal",
            "fragility": "normal",
            "stiffness": "medium",
            "grip_force_multiplier": 0.8,
            "lift_speed_multiplier": 1.2
        },
        "metal": {
            "mass_category": "heavy",
            "friction": "high",
            "fragility": "robust",
            "stiffness": "high",
            "grip_force_multiplier": 1.4,
            "lift_speed_multiplier": 0.7
        },
        "glass": {
            "mass_category": "medium",
            "friction": "low",
            "fragility": "fragile",
            "stiffness": "high",
            "grip_force_multiplier": 0.6,
            "lift_speed_multiplier": 0.5
        },
        "wood": {
            "mass_category": "medium",
            "friction": "high",
            "fragility": "normal",
            "stiffness": "medium",
            "grip_force_multiplier": 1.0,
            "lift_speed_multiplier": 1.0
        },
        "rubber": {
            "mass_category": "light",
            "friction": "very_high",
            "fragility": "robust",
            "stiffness": "low",
            "grip_force_multiplier": 0.7,
            "lift_speed_multiplier": 1.3
        },
        "ceramic": {
            "mass_category": "medium",
            "friction": "normal",
            "fragility": "fragile",
            "stiffness": "high",
            "grip_force_multiplier": 0.65,
            "lift_speed_multiplier": 0.6
        },
        "fabric": {
            "mass_category": "very_light",
            "friction": "normal",
            "fragility": "robust",
            "stiffness": "very_low",
            "grip_force_multiplier": 0.5,
            "lift_speed_multiplier": 1.5
        }
    }

    @classmethod
    def generate_variations(cls, base_params: ControlParameters) -> List[Dict[str, Any]]:
        """물리 속성 변형 생성"""
        variations = []

        for material, props in cls.MATERIAL_PROPERTIES.items():
            # 제어 파라미터 조정
            adjusted_params = ControlParameters(
                grip_force=round(base_params.grip_force * props["grip_force_multiplier"], 2),
                lift_speed=round(base_params.lift_speed * props["lift_speed_multiplier"], 2),
                approach_angle=base_params.approach_angle,
                contact_force=round(base_params.contact_force * props["grip_force_multiplier"], 2),
                safety_margin=base_params.safety_margin * (1.5 if props["fragility"] == "fragile" else 1.0)
            )

            variations.append({
                "material": material,
                "physical_properties": props,
                "control_parameters": adjusted_params
            })

        return variations


class ReasoningGenerator:
    """추론 과정 생성기"""

    REASONING_TEMPLATES = {
        "plastic": "The {object} is likely made of plastic, which is lightweight and has normal friction. I recommend {grip_force} N grip force to securely hold it without deformation. Lift speed of {lift_speed} m/s balances efficiency and safety.",

        "metal": "Given the metallic nature of the {object}, I estimate higher mass and friction. Increased grip force ({grip_force} N) ensures secure grasp despite the weight. Slower lift speed ({lift_speed} m/s) maintains stability.",

        "glass": "The {object} appears to be glass, which is fragile and has low friction. I apply reduced grip force ({grip_force} N) to prevent breakage and slow lift speed ({lift_speed} m/s) for maximum safety. Safety margin increased to {safety_margin}.",

        "wood": "This wooden {object} has medium mass and high friction. Standard grip force ({grip_force} N) and lift speed ({lift_speed} m/s) are appropriate for reliable manipulation.",

        "rubber": "The rubber {object} is lightweight with very high friction. Lower grip force ({grip_force} N) is sufficient due to excellent surface grip. Faster lift speed ({lift_speed} m/s) is safe given the material's resilience.",

        "ceramic": "Ceramic materials are fragile despite medium mass. I reduce grip force to {grip_force} N and lift speed to {lift_speed} m/s to minimize breakage risk. Safety margin set to {safety_margin}.",

        "fabric": "The fabric {object} is very lightweight and flexible. Minimal grip force ({grip_force} N) prevents damage while faster lift speed ({lift_speed} m/s) is safe due to the material's robustness."
    }

    @classmethod
    def generate(cls, material: str, object_name: str, control_params: ControlParameters) -> str:
        """추론 과정 생성"""
        template = cls.REASONING_TEMPLATES.get(material, cls.REASONING_TEMPLATES["plastic"])

        return template.format(
            object=object_name,
            grip_force=control_params.grip_force,
            lift_speed=control_params.lift_speed,
            safety_margin=control_params.safety_margin
        )


class LLMTrainingDataGenerator:
    """LLM 학습 데이터 생성기 (메인)"""

    def __init__(self, episodes_dir: str = "converted_episodes"):
        self.episodes_dir = Path(episodes_dir)
        self.command_paraphraser = CommandParaphraser()
        self.physics_generator = PhysicsVariationGenerator()
        self.reasoning_generator = ReasoningGenerator()

    def load_episodes(self) -> List[Dict[str, Any]]:
        """변환된 DROID 에피소드 로드"""
        # integrated_pipeline_results.json에서 LLM-First 결과 로드
        with open('integrated_pipeline_results.json', 'r') as f:
            results = json.load(f)

        return results['detailed_results']

    def extract_object_name(self, command: str) -> str:
        """명령어에서 객체 이름 추출"""
        # 간단한 추출 로직 (실제로는 더 정교한 NLP 필요)
        if "bottle" in command.lower():
            return "bottle"
        elif "can" in command.lower():
            return "can"
        elif "block" in command.lower():
            return "block"
        elif "cup" in command.lower():
            return "cup"
        elif "ball" in command.lower():
            return "ball"
        elif "mug" in command.lower():
            return "mug"
        elif "bag" in command.lower():
            return "bag"
        else:
            return "object"

    def generate_training_sample(
        self,
        command: str,
        material: str,
        physical_props: Dict[str, Any],
        control_params: ControlParameters
    ) -> Dict[str, Any]:
        """단일 학습 샘플 생성"""

        object_name = self.extract_object_name(command)
        reasoning = self.reasoning_generator.generate(material, object_name, control_params)

        # Friction coefficient 매핑
        friction_map = {
            "low": "0.2-0.3 estimated",
            "normal": "0.4-0.6 estimated",
            "high": "0.6-0.8 estimated",
            "very_high": "0.8-1.0 estimated"
        }

        sample = {
            "instruction": "You are a physics-aware robot control system trained on DROID dataset. Analyze the natural language command and extract physical properties, then generate control parameters for Franka Panda robot.",

            "input": command,

            "output": {
                "physical_analysis": {
                    "material_inference": f"{material}",
                    "mass_category": physical_props["mass_category"],
                    "friction_coefficient": friction_map.get(physical_props["friction"], "0.4-0.6 estimated"),
                    "fragility": physical_props["fragility"],
                    "stiffness": physical_props["stiffness"],
                    "confidence": round(random.uniform(0.75, 0.95), 2)
                },

                "control_parameters": {
                    "grip_force": control_params.grip_force,
                    "lift_speed": control_params.lift_speed,
                    "approach_angle": control_params.approach_angle,
                    "contact_force": control_params.contact_force,
                    "safety_margin": round(control_params.safety_margin, 2)
                },

                "reasoning": reasoning,

                "affordance_assessment": {
                    "success_probability": round(random.uniform(0.80, 0.95), 2),
                    "risk_factors": [] if physical_props["fragility"] != "fragile" else ["fragile_material"],
                    "recommended_approach": "careful_approach" if physical_props["fragility"] == "fragile" else "standard_confident_approach"
                }
            }
        }

        return sample

    def generate_dataset(self, samples_per_episode: int = 10) -> List[Dict[str, Any]]:
        """전체 학습 데이터셋 생성"""

        episodes = self.load_episodes()
        training_data = []

        print(f"🔄 Loaded {len(episodes)} episodes")
        print(f"🎯 Target: {len(episodes) * samples_per_episode} training samples\n")

        for episode_idx, episode in enumerate(episodes):
            base_command = episode['command']
            base_control = episode['llm_first_result']['control_parameters']

            print(f"📊 Episode {episode_idx}: {base_command}")

            # 기본 제어 파라미터 객체 생성
            base_params = ControlParameters(
                grip_force=base_control['grip_force'],
                lift_speed=base_control['lift_speed'],
                approach_angle=base_control['approach_angle'],
                contact_force=base_control['contact_force'],
                safety_margin=base_control['safety_margin']
            )

            # 1. 명령어 변형 생성
            command_variations = self.command_paraphraser.generate_variations(
                base_command,
                num_variations=3
            )

            # 2. 물리 속성 변형 생성
            physics_variations = self.physics_generator.generate_variations(base_params)

            # 3. 조합하여 학습 샘플 생성
            episode_samples = 0
            for cmd_var in command_variations:
                for phys_var in physics_variations:
                    if episode_samples >= samples_per_episode:
                        break

                    sample = self.generate_training_sample(
                        command=cmd_var,
                        material=phys_var["material"],
                        physical_props=phys_var["physical_properties"],
                        control_params=phys_var["control_parameters"]
                    )

                    training_data.append(sample)
                    episode_samples += 1

                if episode_samples >= samples_per_episode:
                    break

            print(f"  ✅ Generated {episode_samples} samples\n")

        print(f"🎉 Total training samples generated: {len(training_data)}")
        return training_data

    def save_dataset(self, dataset: List[Dict[str, Any]], output_path: str = "llm_training_dataset.json"):
        """데이터셋 저장"""

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        print(f"💾 Dataset saved to: {output_path}")

        # 통계 정보 저장
        stats = {
            "total_samples": len(dataset),
            "materials_covered": list(set(s['output']['physical_analysis']['material_inference'] for s in dataset)),
            "avg_confidence": sum(s['output']['physical_analysis']['confidence'] for s in dataset) / len(dataset),
            "format_version": "1.0.0",
            "purpose": "LLM fine-tuning for physical domain robot control"
        }

        with open(output_path.replace('.json', '_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)

        return stats


def main():
    """메인 실행 함수"""

    print("="*60)
    print("🤖 LLM Training Data Generator")
    print("="*60)
    print()

    generator = LLMTrainingDataGenerator()

    # 데이터셋 생성 (에피소드당 20개 샘플 = 총 60개)
    dataset = generator.generate_dataset(samples_per_episode=20)

    # 저장
    stats = generator.save_dataset(dataset)

    print("\n" + "="*60)
    print("📊 Dataset Statistics")
    print("="*60)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Materials covered: {', '.join(stats['materials_covered'])}")
    print(f"Average confidence: {stats['avg_confidence']:.2f}")
    print(f"\n✅ Ready for LLM fine-tuning!")


if __name__ == "__main__":
    main()
