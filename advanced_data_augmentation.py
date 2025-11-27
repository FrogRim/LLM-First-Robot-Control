"""
Advanced Data Augmentation System
혼합 방식: 실제 에피소드 확장 + 공격적 증강

목표: 300-500개 고품질 학습 샘플
전략:
  1. 기존 3개 에피소드를 변형하여 10개로 확장
  2. 각 에피소드당 40-50개 샘플 생성
  3. 다양한 증강 기법 적용
"""

import json
import random
from typing import List, Dict, Any
from copy import deepcopy
from pathlib import Path


class EpisodeExpander:
    """기존 에피소드를 변형하여 새로운 에피소드 생성"""

    @staticmethod
    def add_noise_to_trajectory(trajectory: List[List[float]], noise_level: float = 0.05) -> List[List[float]]:
        """궤적에 노이즈 추가"""
        noisy_trajectory = []
        for pose in trajectory:
            noisy_pose = [
                p + random.gauss(0, noise_level * abs(p)) for p in pose
            ]
            noisy_trajectory.append(noisy_pose)
        return noisy_trajectory

    @staticmethod
    def scale_trajectory(trajectory: List[List[float]], scale_factor: float = 1.0) -> List[List[float]]:
        """궤적 스케일 조정"""
        return [[p * scale_factor for p in pose] for pose in trajectory]

    @staticmethod
    def time_warp_trajectory(trajectory: List[List[float]], warp_factor: float = 1.2) -> List[List[float]]:
        """궤적 시간 왜곡 (속도 변화)"""
        original_len = len(trajectory)
        target_len = int(original_len * warp_factor)

        if target_len <= 1:
            return trajectory

        # numpy 없이 linspace 구현
        step = (original_len - 1) / (target_len - 1) if target_len > 1 else 0
        indices = [i * step for i in range(target_len)]
        warped = []

        for idx in indices:
            lower = int(idx)
            upper = min(int(idx) + 1, original_len - 1)
            alpha = idx - lower

            if lower == upper:
                warped.append(trajectory[lower])
            else:
                interpolated = [
                    (1 - alpha) * trajectory[lower][i] + alpha * trajectory[upper][i]
                    for i in range(len(trajectory[0]))
                ]
                warped.append(interpolated)

        return warped

    def expand_episodes(self, base_episodes: List[Dict], target_count: int = 10) -> List[Dict]:
        """에피소드 확장"""
        expanded = deepcopy(base_episodes)

        transformations = [
            ("noise_small", lambda t: self.add_noise_to_trajectory(t, 0.03)),
            ("noise_medium", lambda t: self.add_noise_to_trajectory(t, 0.05)),
            ("noise_large", lambda t: self.add_noise_to_trajectory(t, 0.08)),
            ("scale_up", lambda t: self.scale_trajectory(t, 1.05)),
            ("scale_down", lambda t: self.scale_trajectory(t, 0.95)),
            ("time_faster", lambda t: self.time_warp_trajectory(t, 0.8)),
            ("time_slower", lambda t: self.time_warp_trajectory(t, 1.2)),
        ]

        episode_id = len(base_episodes)

        while len(expanded) < target_count:
            # 랜덤하게 베이스 에피소드 선택
            base_ep = random.choice(base_episodes)
            transform_name, transform_fn = random.choice(transformations)

            # 새 에피소드 생성
            new_ep = deepcopy(base_ep)
            new_ep['episode_metadata']['id'] = f"genesis_droid_episode_{episode_id:03d}"
            new_ep['episode_metadata']['source'] = f"augmented_from_{base_ep['episode_metadata']['id']}_{transform_name}"

            # 궤적 변형
            if 'trajectory_data' in new_ep and 'joint_positions' in new_ep['trajectory_data']:
                new_ep['trajectory_data']['joint_positions'] = transform_fn(
                    new_ep['trajectory_data']['joint_positions']
                )

            expanded.append(new_ep)
            episode_id += 1

        return expanded[:target_count]


class AdvancedCommandGenerator:
    """고급 명령어 생성기 - 더 다양한 명령어 패턴"""

    # 동작 동사 확장
    ACTION_VERBS = {
        'pick': ['Pick up', 'Grasp', 'Lift', 'Take', 'Grab', 'Hold', 'Secure', 'Retrieve', 'Collect', 'Acquire'],
        'place': ['place', 'put', 'set', 'position', 'move', 'transfer', 'relocate', 'deposit', 'arrange', 'store'],
        'push': ['push', 'slide', 'shove', 'nudge', 'press'],
        'pull': ['pull', 'drag', 'draw', 'tug', 'haul'],
    }

    # 객체 확장
    OBJECTS = [
        # 기본
        'object', 'item', 'target object',
        # 재료별
        'plastic bottle', 'plastic cup', 'plastic container', 'plastic toy',
        'metal can', 'metal box', 'metal tool', 'metal plate',
        'glass cup', 'glass bottle', 'glass jar', 'glass vase',
        'wooden block', 'wooden box', 'wooden toy', 'wooden piece',
        'rubber ball', 'rubber grip', 'rubber pad', 'rubber ring',
        'ceramic mug', 'ceramic plate', 'ceramic bowl', 'ceramic tile',
        'fabric bag', 'fabric cloth', 'fabric pouch', 'fabric cover',
        # 크기/무게 수식어
        'small object', 'large object', 'heavy object', 'light object',
        'thin object', 'thick object', 'flat object', 'round object',
    ]

    # 목적지 확장
    DESTINATIONS = [
        'container', 'box', 'basket', 'bin', 'tray', 'platform', 'holder', 'rack',
        'shelf', 'table', 'stand', 'slot', 'compartment', 'area', 'zone', 'position',
    ]

    # 방식 부사구
    MANNER_ADVERBS = [
        '', 'carefully', 'gently', 'slowly', 'quickly', 'firmly', 'smoothly',
        'cautiously', 'steadily', 'precisely', 'delicately', 'securely',
    ]

    # 제약 조건
    CONSTRAINTS = [
        '',
        'without touching other objects',
        'while maintaining balance',
        'without tilting',
        'keeping it upright',
        'avoiding obstacles',
        'following the shortest path',
        'at constant speed',
    ]

    @classmethod
    def generate_complex_command(cls) -> str:
        """복잡한 명령어 생성"""
        action = random.choice(cls.ACTION_VERBS['pick'])
        obj = random.choice(cls.OBJECTS)
        manner = random.choice(cls.MANNER_ADVERBS)
        place_verb = random.choice(cls.ACTION_VERBS['place'])
        dest = random.choice(cls.DESTINATIONS)
        constraint = random.choice(cls.CONSTRAINTS)

        # 템플릿 조합
        templates = [
            f"{action} the {obj} and {place_verb} it in the {dest}",
            f"{manner} {action.lower()} the {obj} and {place_verb} it in the {dest}" if manner else f"{action} the {obj} and {place_verb} it in the {dest}",
            f"{action} the {obj} {constraint} and {place_verb} it in the {dest}" if constraint else f"{action} the {obj} and {place_verb} it in the {dest}",
            f"{manner} {action.lower()} the {obj} {constraint} and {place_verb} it in the {dest}" if manner and constraint else f"{action} the {obj} and {place_verb} it in the {dest}",
        ]

        return random.choice(templates)

    @classmethod
    def generate_variations(cls, base_command: str, num_variations: int = 10) -> List[str]:
        """명령어 변형 생성"""
        variations = [base_command]

        for _ in range(num_variations - 1):
            variations.append(cls.generate_complex_command())

        return variations


class EnhancedPhysicsVariationGenerator:
    """향상된 물리 속성 변형 생성기"""

    MATERIAL_PROPERTIES = {
        "plastic": {
            "variants": ["light_plastic", "heavy_plastic", "rigid_plastic", "flexible_plastic"],
            "mass_range": ["very_light", "light", "medium"],
            "friction_range": ["low", "normal", "high"],
            "fragility_range": ["robust", "normal"],
            "stiffness_range": ["low", "medium", "high"],
        },
        "metal": {
            "variants": ["aluminum", "steel", "copper", "iron"],
            "mass_range": ["medium", "heavy", "very_heavy"],
            "friction_range": ["normal", "high", "very_high"],
            "fragility_range": ["robust"],
            "stiffness_range": ["high", "very_high"],
        },
        "glass": {
            "variants": ["thin_glass", "thick_glass", "tempered_glass"],
            "mass_range": ["light", "medium"],
            "friction_range": ["low", "normal"],
            "fragility_range": ["fragile", "very_fragile"],
            "stiffness_range": ["high", "very_high"],
        },
        "wood": {
            "variants": ["softwood", "hardwood", "plywood", "balsa"],
            "mass_range": ["light", "medium", "heavy"],
            "friction_range": ["high", "very_high"],
            "fragility_range": ["normal", "robust"],
            "stiffness_range": ["medium", "high"],
        },
        "rubber": {
            "variants": ["soft_rubber", "hard_rubber", "foam_rubber"],
            "mass_range": ["very_light", "light"],
            "friction_range": ["very_high"],
            "fragility_range": ["robust"],
            "stiffness_range": ["very_low", "low"],
        },
        "ceramic": {
            "variants": ["porcelain", "stoneware", "earthenware"],
            "mass_range": ["medium", "heavy"],
            "friction_range": ["normal", "high"],
            "fragility_range": ["fragile", "very_fragile"],
            "stiffness_range": ["high", "very_high"],
        },
        "fabric": {
            "variants": ["cotton", "polyester", "silk", "canvas"],
            "mass_range": ["very_light", "light"],
            "friction_range": ["normal", "high"],
            "fragility_range": ["robust"],
            "stiffness_range": ["very_low", "low"],
        },
    }

    @classmethod
    def generate_detailed_variations(cls, base_material: str) -> List[Dict[str, Any]]:
        """세부 물리 속성 변형 생성"""
        if base_material not in cls.MATERIAL_PROPERTIES:
            base_material = "plastic"

        props = cls.MATERIAL_PROPERTIES[base_material]
        variations = []

        for variant in props["variants"]:
            for mass in props["mass_range"]:
                for friction in props["friction_range"]:
                    for fragility in props["fragility_range"]:
                        for stiffness in props["stiffness_range"]:
                            variations.append({
                                "material": base_material,
                                "variant": variant,
                                "mass_category": mass,
                                "friction": friction,
                                "fragility": fragility,
                                "stiffness": stiffness,
                            })

        return variations


class AdvancedDataAugmentationPipeline:
    """고급 데이터 증강 파이프라인"""

    def __init__(self):
        self.episode_expander = EpisodeExpander()
        self.command_generator = AdvancedCommandGenerator()
        self.physics_generator = EnhancedPhysicsVariationGenerator()

    def load_base_episodes(self) -> List[Dict]:
        """기존 변환된 에피소드 로드"""
        episodes = []
        for i in range(3):
            path = f"converted_episodes/genesis_droid_episode_00{i}.json"
            try:
                with open(path, 'r') as f:
                    episodes.append(json.load(f))
            except:
                pass
        return episodes

    def load_integrated_results(self) -> List[Dict]:
        """통합 파이프라인 결과 로드"""
        try:
            with open('integrated_pipeline_results.json', 'r') as f:
                results = json.load(f)
                return results['detailed_results']
        except:
            return []

    def generate_training_sample(
        self,
        episode: Dict,
        command: str,
        physics_variation: Dict,
        base_control_params: Dict
    ) -> Dict[str, Any]:
        """학습 샘플 생성"""

        # 물리 속성 기반 제어 파라미터 조정
        mass_multipliers = {
            "very_light": 0.5, "light": 0.7, "medium": 1.0,
            "heavy": 1.3, "very_heavy": 1.6
        }

        fragility_multipliers = {
            "robust": 1.0, "normal": 0.8,
            "fragile": 0.6, "very_fragile": 0.4
        }

        mass_mult = mass_multipliers.get(physics_variation["mass_category"], 1.0)
        frag_mult = fragility_multipliers.get(physics_variation["fragility"], 1.0)

        adjusted_grip = base_control_params["grip_force"] * mass_mult * frag_mult
        adjusted_speed = base_control_params["lift_speed"] * (1.0 / mass_mult) * frag_mult
        adjusted_safety = base_control_params["safety_margin"] * (2.0 if physics_variation["fragility"] in ["fragile", "very_fragile"] else 1.0)

        # Friction coefficient 매핑
        friction_map = {
            "low": "0.2-0.3", "normal": "0.4-0.6",
            "high": "0.6-0.8", "very_high": "0.8-1.0"
        }

        # Reasoning 생성
        reasoning = self.generate_reasoning(
            physics_variation,
            adjusted_grip,
            adjusted_speed
        )

        sample = {
            "instruction": "You are a physics-aware robot control system trained on DROID dataset. Analyze the natural language command and extract physical properties, then generate control parameters for Franka Panda robot.",
            "input": command,
            "output": {
                "physical_analysis": {
                    "material_inference": f"{physics_variation['material']} ({physics_variation['variant']})",
                    "mass_category": physics_variation["mass_category"],
                    "friction_coefficient": f"{friction_map.get(physics_variation['friction'], '0.4-0.6')} estimated",
                    "fragility": physics_variation["fragility"],
                    "stiffness": physics_variation["stiffness"],
                    "confidence": round(random.uniform(0.75, 0.95), 2)
                },
                "control_parameters": {
                    "grip_force": round(adjusted_grip, 2),
                    "lift_speed": round(adjusted_speed, 2),
                    "approach_angle": base_control_params.get("approach_angle", 0.0),
                    "contact_force": round(adjusted_grip * 0.6, 2),
                    "safety_margin": round(adjusted_safety, 2)
                },
                "reasoning": reasoning,
                "affordance_assessment": {
                    "success_probability": round(random.uniform(0.75, 0.95), 2),
                    "risk_factors": ["fragile_material"] if physics_variation["fragility"] in ["fragile", "very_fragile"] else [],
                    "recommended_approach": "careful_approach" if physics_variation["fragility"] in ["fragile", "very_fragile"] else "standard_confident_approach"
                }
            }
        }

        return sample

    def generate_reasoning(self, physics: Dict, grip: float, speed: float) -> str:
        """추론 과정 생성"""
        material = physics["material"]
        variant = physics["variant"]
        mass = physics["mass_category"]
        fragility = physics["fragility"]

        templates = {
            "plastic": f"The {variant} object is {mass} with {fragility} characteristics. I recommend {grip}N grip force for secure handling without deformation. Lift speed of {speed} m/s balances efficiency and safety.",
            "metal": f"Given the {variant} nature, I estimate {mass} mass. The {grip}N grip force ensures secure grasp despite weight. Slower lift speed ({speed} m/s) maintains stability.",
            "glass": f"The {variant} material is {fragility}. I apply reduced grip force ({grip}N) to prevent breakage and slow lift speed ({speed} m/s) for maximum safety.",
            "wood": f"This {variant} object has {mass} mass. Standard grip force ({grip}N) and lift speed ({speed} m/s) are appropriate for reliable manipulation.",
            "rubber": f"The {variant} material is {mass} with excellent surface grip. Lower grip force ({grip}N) is sufficient. Faster lift speed ({speed} m/s) is safe given material resilience.",
            "ceramic": f"{variant.capitalize()} materials are {fragility} despite {mass} mass. I reduce grip force to {grip}N and lift speed to {speed} m/s to minimize breakage risk.",
            "fabric": f"The {variant} material is {mass} and flexible. Minimal grip force ({grip}N) prevents damage while {speed} m/s lift speed is safe due to robustness."
        }

        return templates.get(material, templates["plastic"])

    def generate_augmented_dataset(
        self,
        samples_per_episode: int = 40,
        target_episodes: int = 10
    ) -> List[Dict[str, Any]]:
        """증강된 데이터셋 생성"""

        print("="*70)
        print("🚀 Advanced Data Augmentation Pipeline")
        print("="*70)
        print()

        # Step 1: 에피소드 확장
        print("📊 Step 1: 에피소드 확장 중...")
        base_episodes = self.load_base_episodes()
        print(f"   기존 에피소드: {len(base_episodes)}개")

        expanded_episodes = self.episode_expander.expand_episodes(base_episodes, target_episodes)
        print(f"   확장 후 에피소드: {len(expanded_episodes)}개")
        print()

        # Step 2: 기존 LLM-First 결과 로드
        integrated_results = self.load_integrated_results()

        # Step 3: 각 에피소드에 대해 샘플 생성
        print("🎯 Step 2: 학습 샘플 생성 중...")
        all_samples = []

        for ep_idx, episode in enumerate(expanded_episodes):
            print(f"   에피소드 {ep_idx + 1}/{len(expanded_episodes)}: ", end="")

            # 베이스 제어 파라미터 (integrated_results에서 가져오거나 기본값)
            if ep_idx < len(integrated_results):
                base_control = integrated_results[ep_idx % len(integrated_results)]['llm_first_result']['control_parameters']
            else:
                base_control = {
                    "grip_force": 0.5,
                    "lift_speed": 0.5,
                    "approach_angle": 0.0,
                    "contact_force": 0.3,
                    "safety_margin": 0.8
                }

            # 명령어 생성
            commands = self.command_generator.generate_variations(
                f"Pick up the object and place it in the container {ep_idx}",
                num_variations=min(10, samples_per_episode // 7 + 1)
            )

            episode_samples = 0

            for material in ["plastic", "metal", "glass", "wood", "rubber", "ceramic", "fabric"]:
                # 재료별 세부 변형
                physics_variations = self.physics_generator.generate_detailed_variations(material)

                # 샘플링
                sampled_physics = random.sample(physics_variations, min(len(physics_variations), samples_per_episode // 7))

                for physics_var in sampled_physics:
                    if episode_samples >= samples_per_episode:
                        break

                    command = random.choice(commands)
                    sample = self.generate_training_sample(
                        episode, command, physics_var, base_control
                    )
                    all_samples.append(sample)
                    episode_samples += 1

                if episode_samples >= samples_per_episode:
                    break

            print(f"{episode_samples}개 샘플 생성 ✅")

        print()
        print(f"🎉 총 {len(all_samples)}개 학습 샘플 생성 완료!")

        return all_samples

    def save_augmented_dataset(self, dataset: List[Dict], output_path: str = "llm_training_dataset_augmented.json"):
        """증강된 데이터셋 저장"""

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        # 통계 생성
        materials = {}
        for sample in dataset:
            mat = sample['output']['physical_analysis']['material_inference'].split()[0]
            materials[mat] = materials.get(mat, 0) + 1

        stats = {
            "total_samples": len(dataset),
            "materials_distribution": materials,
            "avg_confidence": sum(s['output']['physical_analysis']['confidence'] for s in dataset) / len(dataset),
            "format_version": "2.0.0",
            "augmentation_method": "hybrid (episode expansion + aggressive augmentation)",
            "purpose": "LLM fine-tuning for physical domain robot control (ENHANCED)"
        }

        with open(output_path.replace('.json', '_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"💾 데이터셋 저장: {output_path}")
        print(f"📊 통계 저장: {output_path.replace('.json', '_stats.json')}")

        return stats


def main():
    """메인 실행"""

    pipeline = AdvancedDataAugmentationPipeline()

    # 목표: 400개 샘플 (10 episodes × 40 samples)
    dataset = pipeline.generate_augmented_dataset(
        samples_per_episode=40,
        target_episodes=10
    )

    # 저장
    stats = pipeline.save_augmented_dataset(dataset)

    print()
    print("="*70)
    print("📊 최종 통계")
    print("="*70)
    print(f"총 샘플 수: {stats['total_samples']}")
    print(f"평균 신뢰도: {stats['avg_confidence']:.2f}")
    print(f"\n재료별 분포:")
    for material, count in sorted(stats['materials_distribution'].items()):
        print(f"  - {material}: {count}개 ({count/stats['total_samples']*100:.1f}%)")
    print()
    print("✅ 고품질 증강 데이터셋 생성 완료!")
    print(f"   → Qwen2.5-14B 학습 준비 완료! 🚀")


if __name__ == "__main__":
    main()
