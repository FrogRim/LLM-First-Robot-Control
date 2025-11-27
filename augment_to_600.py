"""
데이터 증강: 350개 → 600개로 확대
"""

import sys
sys.path.insert(0, '/root/gen')

from advanced_data_augmentation import AdvancedDataAugmentationPipeline

def main():
    """600개 샘플 생성 (15 episodes × 40 samples)"""
    
    pipeline = AdvancedDataAugmentationPipeline()
    
    print("="*70)
    print("🚀 데이터 증강 시작: 목표 600개 샘플")
    print("   - 에피소드 수: 15개")
    print("   - 에피소드당 샘플: 40개")
    print("="*70)
    print()
    
    # 600개 샘플 생성
    dataset = pipeline.generate_augmented_dataset(
        samples_per_episode=40,
        target_episodes=15  # 10 → 15로 증가
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
    print("✅ 600개 샘플 생성 완료! (기존 350개 → 600개)")
    print(f"   → 71% 증가! 🚀")
    print()

if __name__ == "__main__":
    main()

