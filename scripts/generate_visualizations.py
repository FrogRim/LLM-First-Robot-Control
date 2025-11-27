#!/usr/bin/env python3
"""
논문용 시각화 생성
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 스타일 설정
sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_training_loss():
    """학습 손실 곡선"""
    epochs = [1, 2, 3, 4]
    train_loss = [1.234, 0.723, 0.489, 0.334]
    eval_loss = [0.912, 0.589, 0.412, 0.325]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, train_loss, marker='o', linewidth=2, markersize=8, label='Training Loss')
    ax.plot(epochs, eval_loss, marker='s', linewidth=2, markersize=8, label='Evaluation Loss')
    
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss (Cross-Entropy)', fontsize=14, fontweight='bold')
    ax.set_title('Training and Evaluation Loss Over Epochs', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 최종 값 표시
    ax.annotate(f'Final: {train_loss[-1]:.3f}', 
                xy=(epochs[-1], train_loss[-1]), 
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    ax.annotate(f'Final: {eval_loss[-1]:.3f}', 
                xy=(epochs[-1], eval_loss[-1]), 
                xytext=(10, -20), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('figures/training_loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 학습 손실 곡선 저장: figures/training_loss_curve.png")


def plot_baseline_comparison():
    """Baseline 비교 막대 그래프"""
    categories = ['JSON Parsing\n(Material Explicit)', 'Material Recognition\n(Implicit)', 'Avg Confidence']
    base_scores = [100, 57.1, 0.850 * 100]
    finetuned_scores = [100, 71.4, 0.853 * 100]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar(x - width/2, base_scores, width, label='Base Model', 
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, finetuned_scores, width, label='Fine-tuned Model',
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
    ax.set_title('Baseline Comparison: Base Model vs Fine-tuned Model', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.legend(fontsize=12, loc='lower right')
    ax.set_ylim([0, 110])
    
    # 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%' if height != 85.0 and height != 85.3 else f'{height/100:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold')
    
    # 개선 화살표 (암묵적 추론)
    ax.annotate('', xy=(1 + width/2, 71.4), xytext=(1 - width/2, 57.1),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.text(1, 64, '+14.3%p', fontsize=11, fontweight='bold', 
            ha='center', color='green',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('figures/baseline_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Baseline 비교 그래프 저장: figures/baseline_comparison.png")


def plot_generalization_details():
    """일반화 테스트 상세 비교"""
    test_cases = [
        'Transparent\nCup',
        'Heavy Shiny\nTool', 
        'Fragile\nDish',
        'Lightweight\nContainer',
        'Soft\nCloth',
        'Solid\nBlock',
        'Flexible\nTube'
    ]
    
    base_correct = [1, 1, 1, 1, 0, 0, 0]  # 4/7
    finetuned_correct = [1, 1, 1, 0, 1, 0, 1]  # 5/7
    
    x = np.arange(len(test_cases))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    bars1 = ax.bar(x - width/2, base_correct, width, label='Base Model',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, finetuned_correct, width, label='Fine-tuned Model',
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Correct (1) / Incorrect (0)', fontsize=14, fontweight='bold')
    ax.set_title('Implicit Material Inference Test: Detailed Results', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(test_cases, fontsize=11)
    ax.legend(fontsize=12, loc='upper right')
    ax.set_ylim([0, 1.3])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Incorrect', 'Correct'])
    
    # 정답/오답 표시
    for i, (b, f) in enumerate(zip(base_correct, finetuned_correct)):
        if b == 1 and f == 1:
            ax.text(i, 1.15, '✓✓', ha='center', fontsize=16, color='green')
        elif b == 1 and f == 0:
            ax.text(i, 1.15, '✓✗', ha='center', fontsize=16, color='orange')
        elif b == 0 and f == 1:
            ax.text(i, 1.15, '✗✓', ha='center', fontsize=16, color='blue')
        else:
            ax.text(i, 1.15, '✗✗', ha='center', fontsize=16, color='red')
    
    # 정확도 표시
    base_acc = sum(base_correct) / len(base_correct) * 100
    finetuned_acc = sum(finetuned_correct) / len(finetuned_correct) * 100
    
    ax.text(0.02, 0.98, f'Base Accuracy: {base_acc:.1f}% (4/7)', 
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#3498db', alpha=0.3))
    
    ax.text(0.02, 0.88, f'Fine-tuned Accuracy: {finetuned_acc:.1f}% (5/7)', 
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('figures/generalization_test_details.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 일반화 테스트 상세 그래프 저장: figures/generalization_test_details.png")


def plot_inference_time_comparison():
    """추론 시간 비교"""
    models = ['Base Model', 'Fine-tuned Model\n(v2, 256 tokens)', 'Fine-tuned Model\n(optimized, 128 tokens)']
    times = [9.3, 30.5, 15.6]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    bars = ax.bar(models, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Inference Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_title('Average Inference Time Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim([0, 35])
    
    # 값 표시
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
    
    # 개선율 표시
    ax.annotate('', xy=(2, 15.6), xytext=(1, 30.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.text(1.5, 23, '-39%', fontsize=12, fontweight='bold', 
            ha='center', color='green',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    
    # 목표선
    ax.axhline(y=0.2, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Target: 200ms')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/inference_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 추론 시간 비교 그래프 저장: figures/inference_time_comparison.png")


def plot_dataset_growth():
    """데이터셋 증가 추이"""
    versions = ['v1\n(Initial)', 'v2\n(Augmented)', 'v3\n(Expanded)']
    samples = [60, 350, 525]
    colors = ['#95a5a6', '#3498db', '#2ecc71']
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    bars = ax.bar(versions, samples, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
    ax.set_title('Dataset Growth Over Versions', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim([0, 600])
    
    # 값 표시
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
    
    # 증가율 표시
    ax.annotate('', xy=(1, 350), xytext=(0, 60),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax.text(0.5, 200, '+483%', fontsize=11, fontweight='bold', 
            ha='center', color='blue',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    
    ax.annotate('', xy=(2, 525), xytext=(1, 350),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.text(1.5, 440, '+50%', fontsize=11, fontweight='bold', 
            ha='center', color='green',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('figures/dataset_growth.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 데이터셋 증가 그래프 저장: figures/dataset_growth.png")


def plot_material_distribution():
    """재료별 분포 (v3)"""
    materials = ['Plastic', 'Metal', 'Glass', 'Wood', 'Rubber', 'Ceramic', 'Fabric']
    counts = [75, 75, 75, 75, 75, 75, 75]
    colors = ['#3498db', '#95a5a6', '#9b59b6', '#d35400', '#34495e', '#e67e22', '#1abc9c']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars = ax.bar(materials, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
    ax.set_title('Material Distribution in v3 Dataset (Perfect Balance)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim([0, 90])
    
    # 값 및 비율 표시
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}\n(14.3%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
    
    # 완벽한 균형 강조
    ax.axhline(y=75, color='green', linestyle='--', linewidth=2, alpha=0.5, 
               label='Perfect Balance (75 each)')
    ax.legend(fontsize=11, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('figures/material_distribution_v3.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 재료 분포 그래프 저장: figures/material_distribution_v3.png")


def plot_key_findings_summary():
    """핵심 발견 요약 그래프"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Baseline 비교 (간단 버전)
    models = ['Base', 'Fine-tuned']
    json_rates = [100, 100]
    implicit_rates = [57.1, 71.4]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax1.bar(x - width/2, json_rates, width, label='Explicit (Material Named)', 
            color='#95a5a6', alpha=0.8)
    ax1.bar(x + width/2, implicit_rates, width, label='Implicit (No Material Name)',
            color='#e74c3c', alpha=0.8)
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Baseline Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 110])
    
    # 개선 강조
    ax1.annotate('+14.3%p', xy=(1 + width/2, 71.4), xytext=(5, 5),
                textcoords='offset points', fontsize=11, fontweight='bold',
                color='green', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # 2. Perplexity
    metrics = ['Train', 'Eval']
    perplexity = [1.32, 1.38]
    
    bars = ax2.bar(metrics, perplexity, color=['#2ecc71', '#3498db'], alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Model Perplexity', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 2])
    ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Ideal: 1.0')
    ax2.legend(fontsize=10)
    
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
    
    # 3. 데이터셋 증가
    versions = ['v1', 'v2', 'v3']
    samples = [60, 350, 525]
    
    ax3.plot(versions, samples, marker='o', linewidth=3, markersize=12, color='#2ecc71')
    ax3.fill_between(range(len(versions)), samples, alpha=0.3, color='#2ecc71')
    ax3.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax3.set_title('(C) Dataset Expansion', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, 600])
    ax3.grid(True, alpha=0.3)
    
    for i, (v, s) in enumerate(zip(versions, samples)):
        ax3.annotate(f'{s}',
                    xy=(i, s),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
    
    # 4. 추론 시간 개선
    stages = ['Initial', 'Optimized']
    times = [30.5, 15.6]
    
    bars = ax4.bar(stages, times, color=['#e74c3c', '#2ecc71'], alpha=0.8, edgecolor='black')
    ax4.set_ylabel('Inference Time (seconds)', fontsize=12, fontweight='bold')
    ax4.set_title('(D) Inference Time Optimization', fontsize=14, fontweight='bold')
    ax4.set_ylim([0, 35])
    
    for bar in bars:
        height = bar.get_height()
        ax4.annotate(f'{height:.1f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
    
    # 개선율
    ax4.annotate('', xy=(1, 15.6), xytext=(0, 30.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax4.text(0.5, 23, '-39%', fontsize=11, fontweight='bold',
            ha='center', color='green',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('figures/key_findings_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 핵심 발견 요약 그래프 저장: figures/key_findings_summary.png")


def plot_perplexity_over_epochs():
    """Epoch별 Perplexity 변화"""
    epochs = [1, 2, 3, 4]
    train_loss = [1.234, 0.723, 0.489, 0.334]
    eval_loss = [0.912, 0.589, 0.412, 0.325]
    
    train_perplexity = [np.exp(l) for l in train_loss]
    eval_perplexity = [np.exp(l) for l in eval_loss]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, train_perplexity, marker='o', linewidth=2, markersize=8, 
            label='Training Perplexity', color='#2ecc71')
    ax.plot(epochs, eval_perplexity, marker='s', linewidth=2, markersize=8,
            label='Evaluation Perplexity', color='#3498db')
    
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Perplexity', fontsize=14, fontweight='bold')
    ax.set_title('Perplexity Improvement Over Training', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([1, 3.5])
    
    # 이상적 값 선
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Ideal: 1.0')
    
    # 최종 값 강조
    ax.annotate(f'Final: {train_perplexity[-1]:.2f}', 
                xy=(epochs[-1], train_perplexity[-1]), 
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    
    ax.annotate(f'Final: {eval_perplexity[-1]:.2f}', 
                xy=(epochs[-1], eval_perplexity[-1]), 
                xytext=(10, -20), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('figures/perplexity_over_epochs.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Perplexity 변화 그래프 저장: figures/perplexity_over_epochs.png")


def main():
    """모든 시각화 생성"""
    print("=" * 70)
    print("📊 논문용 시각화 생성 시작")
    print("=" * 70)
    print()
    
    # figures 디렉토리 생성
    Path('figures').mkdir(exist_ok=True)
    
    # 모든 그래프 생성
    plot_training_loss()
    plot_baseline_comparison()
    plot_generalization_details()
    plot_inference_time_comparison()
    plot_dataset_growth()
    plot_perplexity_over_epochs()
    
    print()
    print("=" * 70)
    print("🎉 모든 시각화 생성 완료!")
    print("=" * 70)
    print()
    print("📁 생성된 파일 (figures/ 디렉토리):")
    print("   1. training_loss_curve.png - 학습 손실 곡선")
    print("   2. baseline_comparison.png - Baseline 비교")
    print("   3. generalization_test_details.png - 일반화 테스트 상세")
    print("   4. inference_time_comparison.png - 추론 시간 비교")
    print("   5. dataset_growth.png - 데이터셋 증가")
    print("   6. perplexity_over_epochs.png - Perplexity 변화")
    print()
    print("💡 논문에 삽입 준비 완료!")
    print()


if __name__ == "__main__":
    main()

