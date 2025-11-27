"""
Hugging Face Datasets 포맷 변환기
LLM fine-tuning 라이브러리(Axolotl, TRL, etc.)와 호환되는 포맷으로 변환
"""

import json
from pathlib import Path


def convert_to_alpaca_format(dataset):
    """
    Alpaca/Axolotl 포맷으로 변환
    - instruction: 시스템 프롬프트
    - input: 사용자 입력
    - output: 예상 출력 (JSON 문자열)
    """
    alpaca_format = []

    for sample in dataset:
        alpaca_sample = {
            "instruction": sample["instruction"],
            "input": sample["input"],
            "output": json.dumps(sample["output"], ensure_ascii=False, indent=2)
        }
        alpaca_format.append(alpaca_sample)

    return alpaca_format


def convert_to_sharegpt_format(dataset):
    """
    ShareGPT/ChatML 포맷으로 변환
    - conversations: [{"from": "human/gpt", "value": "..."}]
    """
    sharegpt_format = []

    for sample in dataset:
        conversation = {
            "conversations": [
                {
                    "from": "system",
                    "value": sample["instruction"]
                },
                {
                    "from": "human",
                    "value": sample["input"]
                },
                {
                    "from": "gpt",
                    "value": json.dumps(sample["output"], ensure_ascii=False, indent=2)
                }
            ]
        }
        sharegpt_format.append(conversation)

    return sharegpt_format


def split_train_test(dataset, test_ratio=0.15):
    """
    Train/Test split (85/15)
    """
    import random
    random.seed(42)

    shuffled = dataset.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - test_ratio))

    train = shuffled[:split_idx]
    test = shuffled[split_idx:]

    return train, test


def main():
    print("="*60)
    print("🔄 Converting to Hugging Face Format")
    print("="*60)

    # 원본 데이터셋 로드
    with open('llm_training_dataset.json', 'r') as f:
        dataset = json.load(f)

    print(f"\n📊 Original dataset: {len(dataset)} samples")

    # Train/Test split
    train_data, test_data = split_train_test(dataset, test_ratio=0.15)
    print(f"✂️  Split: {len(train_data)} train / {len(test_data)} test")

    # 1. Alpaca 포맷 (권장)
    print("\n1️⃣  Converting to Alpaca format...")
    train_alpaca = convert_to_alpaca_format(train_data)
    test_alpaca = convert_to_alpaca_format(test_data)

    with open('droid_physics_llm_train_alpaca.json', 'w', encoding='utf-8') as f:
        json.dump(train_alpaca, f, indent=2, ensure_ascii=False)

    with open('droid_physics_llm_test_alpaca.json', 'w', encoding='utf-8') as f:
        json.dump(test_alpaca, f, indent=2, ensure_ascii=False)

    print(f"   ✅ Saved: droid_physics_llm_train_alpaca.json ({len(train_alpaca)} samples)")
    print(f"   ✅ Saved: droid_physics_llm_test_alpaca.json ({len(test_alpaca)} samples)")

    # 2. ShareGPT 포맷 (대화형 모델용)
    print("\n2️⃣  Converting to ShareGPT format...")
    train_sharegpt = convert_to_sharegpt_format(train_data)
    test_sharegpt = convert_to_sharegpt_format(test_data)

    with open('droid_physics_llm_train_sharegpt.json', 'w', encoding='utf-8') as f:
        json.dump(train_sharegpt, f, indent=2, ensure_ascii=False)

    with open('droid_physics_llm_test_sharegpt.json', 'w', encoding='utf-8') as f:
        json.dump(test_sharegpt, f, indent=2, ensure_ascii=False)

    print(f"   ✅ Saved: droid_physics_llm_train_sharegpt.json ({len(train_sharegpt)} samples)")
    print(f"   ✅ Saved: droid_physics_llm_test_sharegpt.json ({len(test_sharegpt)} samples)")

    # 데이터셋 정보 파일 생성
    dataset_info = {
        "dataset_name": "DROID Physics-Aware Robot Control",
        "version": "1.0.0",
        "purpose": "Fine-tuning LLMs for physical domain robot control",
        "base_dataset": "DROID (Distributed Robot Interaction Dataset)",
        "target_robot": "Franka Emika Panda",
        "simulation_environment": "Genesis AI",
        "total_samples": len(dataset),
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "materials_covered": ["plastic", "metal", "glass", "wood", "rubber", "ceramic", "fabric"],
        "output_format": "JSON (physical_analysis + control_parameters + reasoning + affordance)",
        "recommended_models": [
            "mistralai/Mistral-7B-v0.1",
            "meta-llama/Llama-2-7b-hf",
            "microsoft/phi-2"
        ],
        "training_method": "QLoRA (4-bit quantization recommended)",
        "estimated_training_time": "12-18 hours on RTX 3090",
        "formats": {
            "alpaca": "Recommended for instruction tuning (Axolotl, TRL)",
            "sharegpt": "For conversational models (ChatML compatible)"
        }
    }

    with open('dataset_info.json', 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("✅ Conversion Complete!")
    print("="*60)
    print("\n📦 Generated Files:")
    print("   • droid_physics_llm_train_alpaca.json (Alpaca format - Train)")
    print("   • droid_physics_llm_test_alpaca.json (Alpaca format - Test)")
    print("   • droid_physics_llm_train_sharegpt.json (ShareGPT format - Train)")
    print("   • droid_physics_llm_test_sharegpt.json (ShareGPT format - Test)")
    print("   • dataset_info.json (Dataset metadata)")
    print("\n🚀 Ready for LLM fine-tuning with Axolotl/TRL!")


if __name__ == "__main__":
    main()
