import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Ensure project root is on PYTHONPATH when run from elsewhere
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

# Reuse augmentation pipeline from repo root
from advanced_data_augmentation import AdvancedDataAugmentationPipeline
# Reuse converters
from convert_to_hf_format import convert_to_alpaca_format, split_train_test


def build_augmented_dataset(total_episodes: int = 30, samples_per_episode: int = 40) -> List[Dict[str, Any]]:
    pipeline = AdvancedDataAugmentationPipeline()
    dataset = pipeline.generate_augmented_dataset(
        samples_per_episode=samples_per_episode,
        target_episodes=total_episodes,
    )
    # Save raw augmented set for reference
    Path(".").mkdir(parents=True, exist_ok=True)
    with open("llm_training_dataset_augmented.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    return dataset


def to_alpaca_v2(dataset: List[Dict[str, Any]], test_ratio: float = 0.15) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train, test = split_train_test(dataset, test_ratio=test_ratio)
    train_alpaca = convert_to_alpaca_format(train)
    test_alpaca = convert_to_alpaca_format(test)
    with open("droid_physics_llm_train_alpaca_v2.json", "w", encoding="utf-8") as f:
        json.dump(train_alpaca, f, indent=2, ensure_ascii=False)
    with open("droid_physics_llm_test_alpaca_v2.json", "w", encoding="utf-8") as f:
        json.dump(test_alpaca, f, indent=2, ensure_ascii=False)
    return train_alpaca, test_alpaca


def write_dataset_info_v2(total: int, train_count: int, test_count: int):
    info = {
        "dataset_name": "DROID Physics-Aware Robot Control (Enhanced v2)",
        "version": "2.0.0",
        "total_samples": total,
        "train_samples": train_count,
        "test_samples": test_count,
        "augmentation_method": "Hybrid (Episode expansion + Aggressive augmentation)",
        "episodes": None,
        "samples_per_episode": None,
        "unique_commands": None,
        "materials": 7,
        "selected_model": "Qwen/Qwen2.5-14B-Instruct",
        "estimated_training_time": "6-8 hours on RTX 3090",
    }
    with open("dataset_info_v2.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def main():
    print("=" * 70)
    print("📦 Building Enhanced v2 Dataset (≥1,000 samples)")
    print("=" * 70)

    # 1) Build augmented dataset (default: 30×40=1200)
    dataset = build_augmented_dataset(total_episodes=30, samples_per_episode=40)
    total = len(dataset)
    print(f"Total augmented samples: {total}")

    # 2) Convert to Alpaca v2 train/test
    train_v2, test_v2 = to_alpaca_v2(dataset)
    print(f"Saved v2: train={len(train_v2)} / test={len(test_v2)}")

    # 3) Write dataset_info_v2.json
    write_dataset_info_v2(total=total, train_count=len(train_v2), test_count=len(test_v2))
    print("dataset_info_v2.json updated")

    print("\n✅ v2 dataset ready: droid_physics_llm_train_alpaca_v2.json, droid_physics_llm_test_alpaca_v2.json")


if __name__ == "__main__":
    main()


