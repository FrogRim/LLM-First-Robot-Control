#!/usr/bin/env python3
"""
증강된 데이터셋(525개)를 Alpaca 포맷으로 변환
"""

import json
import random

def convert_to_alpaca(dataset, train_ratio=0.85):
    """Alpaca 포맷 변환 및 train/test 분할"""
    alpaca_data = []
    
    for sample in dataset:
        alpaca_sample = {
            "instruction": sample["instruction"],
            "input": sample["input"],
            "output": json.dumps(sample["output"], ensure_ascii=False, indent=2)
        }
        alpaca_data.append(alpaca_sample)
    
    # 셔플 및 분할
    random.shuffle(alpaca_data)
    split_idx = int(len(alpaca_data) * train_ratio)
    
    train_data = alpaca_data[:split_idx]
    test_data = alpaca_data[split_idx:]
    
    return train_data, test_data


def main():
    print("="*70)
    print("🔄 데이터셋 변환: v3 (525개 샘플)")
    print("="*70)
    
    # 증강된 데이터 로드
    with open('llm_training_dataset_augmented.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"\n📊 원본 데이터: {len(dataset)}개 샘플")
    
    # Alpaca 포맷으로 변환 및 분할
    train_data, test_data = convert_to_alpaca(dataset, train_ratio=0.85)
    
    print(f"✂️  분할: {len(train_data)} train / {len(test_data)} test")
    
    # 저장
    with open('droid_physics_llm_train_alpaca_v3.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open('droid_physics_llm_test_alpaca_v3.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print()
    print("✅ Alpaca 포맷 변환 완료!")
    print(f"   • droid_physics_llm_train_alpaca_v3.json ({len(train_data)} samples)")
    print(f"   • droid_physics_llm_test_alpaca_v3.json ({len(test_data)} samples)")
    print()
    print("📈 데이터셋 비교:")
    print(f"   v2: 350개 (train 297, test 53)")
    print(f"   v3: 525개 (train {len(train_data)}, test {len(test_data)})")
    print(f"   증가: +{525-350}개 (+{(525-350)/350*100:.1f}%)")
    print()


if __name__ == "__main__":
    main()

