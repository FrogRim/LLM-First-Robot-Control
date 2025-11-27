#!/usr/bin/env python3
"""
비교 실험 결과 분석 및 시각화
막대그래프, 테이블, LLM 출력 예시 생성
"""

import json
import os
import argparse
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# matplotlib 한글 지원
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# seaborn 스타일 설정
sns.set_style("whitegrid")
sns.set_palette("husl")


class ComparisonVisualizer:
    """비교 실험 결과 시각화 클래스"""

    def __init__(self, results_file: str):
        """
        Args:
            results_file: 실험 결과 JSON 파일 경로
        """
        self.results_file = results_file
        self.data = self._load_results()
        self.output_dir = Path(results_file).parent / "visualizations"
        self.output_dir.mkdir(exist_ok=True)

        print(f"📊 결과 데이터 로드: {results_file}")
        print(f"📁 시각화 저장 위치: {self.output_dir}")

    def _load_results(self) -> Dict[str, Any]:
        """결과 데이터 로드"""
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 결과 파일 로드 실패: {e}")
            return {}

    def create_success_rate_comparison(self, save_path: Optional[str] = None) -> str:
        """성공률 비교 막대그래프 생성"""
        if 'performance_comparison' not in self.data:
            print("⚠️  성능 비교 데이터가 없습니다")
            return ""

        success_rates = self.data['performance_comparison']['success_rates']

        # 데이터 준비
        scenarios = []
        controllers = set()
        data_points = []

        for scenario, controller_data in success_rates.items():
            scenarios.append(scenario.replace('_', ' ').title())
            for controller, metrics in controller_data.items():
                controllers.add(controller)
                data_points.append({
                    'scenario': scenario.replace('_', ' ').title(),
                    'controller': controller,
                    'success_rate': metrics['success_rate'] * 100,  # 백분율로 변환
                    'runs': metrics['runs']
                })

        if not data_points:
            print("⚠️  성공률 데이터가 없습니다")
            return ""

        df = pd.DataFrame(data_points)

        # 그래프 생성
        plt.figure(figsize=(12, 8))

        # 컨트롤러별 색상 매핑
        controller_colors = {
            'Rule-Based': '#2E86AB',
            'RL-Based': '#A23B72',
            'LLM-First': '#F18F01'
        }

        ax = sns.barplot(data=df, x='scenario', y='success_rate', hue='controller',
                        palette=controller_colors)

        # 그래프 꾸미기
        plt.title('Controller Success Rates by Scenario', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Scenario', fontsize=12)
        plt.ylabel('Success Rate (%)', fontsize=12)
        plt.ylim(0, 105)

        # 값 레이블 추가
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', fontsize=10, padding=3)

        plt.legend(title='Controller', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # 저장
        if save_path is None:
            save_path = self.output_dir / "success_rate_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 성공률 비교 그래프 저장: {save_path}")
        return str(save_path)

    def create_efficiency_comparison(self, save_path: Optional[str] = None) -> str:
        """실행 효율성 비교 그래프 생성"""
        if 'performance_comparison' not in self.data:
            print("⚠️  성능 비교 데이터가 없습니다")
            return ""

        efficiency_data = self.data['performance_comparison']['efficiency_metrics']

        # 데이터 준비
        controllers = []
        inference_times = []
        total_times = []

        for controller, metrics in efficiency_data.items():
            controllers.append(controller)

            # 추론 시간 (ms)
            if 'inference_time' in metrics:
                inference_times.append(metrics['inference_time']['mean'])
            else:
                inference_times.append(0)

            # 총 실행 시간 (초)
            if 'total_time' in metrics:
                total_times.append(metrics['total_time']['mean'])
            else:
                total_times.append(0)

        if not controllers:
            print("⚠️  효율성 데이터가 없습니다")
            return ""

        # 그래프 생성
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 추론 시간 그래프
        bars1 = ax1.bar(controllers, inference_times, color=['#2E86AB', '#A23B72', '#F18F01'])
        ax1.set_title('Average Inference Time by Controller', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Time (ms)', fontsize=12)
        ax1.set_xlabel('Controller', fontsize=12)

        # 값 레이블 추가
        for bar, value in zip(bars1, inference_times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(inference_times)*0.02,
                    '.1f', ha='center', va='bottom', fontsize=10)

        # 총 실행 시간 그래프
        bars2 = ax2.bar(controllers, total_times, color=['#2E86AB', '#A23B72', '#F18F01'])
        ax2.set_title('Average Total Execution Time by Controller', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Time (seconds)', fontsize=12)
        ax2.set_xlabel('Controller', fontsize=12)

        # 값 레이블 추가
        for bar, value in zip(bars2, total_times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(total_times)*0.02,
                    '.2f', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        # 저장
        if save_path is None:
            save_path = self.output_dir / "efficiency_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 효율성 비교 그래프 저장: {save_path}")
        return str(save_path)

    def create_failure_analysis_chart(self, save_path: Optional[str] = None) -> str:
        """실패 유형 분석 차트 생성"""
        if 'performance_comparison' not in self.data:
            print("⚠️  성능 비교 데이터가 없습니다")
            return ""

        failure_data = self.data['performance_comparison']['failure_analysis']

        # 데이터 준비
        failure_categories = {}
        controllers = []

        # 모든 컨트롤러의 실패 카테고리 수집
        all_categories = set()
        for controller, analysis in failure_data.items():
            if 'failure_categories' in analysis:
                all_categories.update(analysis['failure_categories'].keys())

        all_categories = sorted(list(all_categories))

        for controller, analysis in failure_data.items():
            controllers.append(controller)
            failure_categories[controller] = {}
            categories = analysis.get('failure_categories', {})

            for category in all_categories:
                failure_categories[controller][category] = categories.get(category, 0)

        if not controllers:
            print("⚠️  실패 분석 데이터가 없습니다")
            return ""

        # 데이터프레임 생성
        df_data = []
        for controller in controllers:
            for category in all_categories:
                df_data.append({
                    'controller': controller,
                    'category': category.replace('_', ' ').title(),
                    'count': failure_categories[controller][category]
                })

        df = pd.DataFrame(df_data)

        # 그래프 생성
        plt.figure(figsize=(12, 8))

        controller_colors = {
            'Rule-Based': '#2E86AB',
            'RL-Based': '#A23B72',
            'LLM-First': '#F18F01'
        }

        ax = sns.barplot(data=df, x='category', y='count', hue='controller',
                        palette=controller_colors)

        # 그래프 꾸미기
        plt.title('Failure Analysis by Category and Controller', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Failure Category', fontsize=12)
        plt.ylabel('Number of Failures', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Controller', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # 저장
        if save_path is None:
            save_path = self.output_dir / "failure_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 실패 분석 차트 저장: {save_path}")
        return str(save_path)

    def create_scenario_performance_heatmap(self, save_path: Optional[str] = None) -> str:
        """시나리오별 성능 히트맵 생성"""
        if 'performance_comparison' not in self.data:
            print("⚠️  성능 비교 데이터가 없습니다")
            return ""

        success_rates = self.data['performance_comparison']['success_rates']

        # 데이터 준비
        scenarios = []
        controllers = set()

        for scenario in success_rates.keys():
            scenarios.append(scenario.replace('_', ' ').title())

        for scenario_data in success_rates.values():
            controllers.update(scenario_data.keys())

        controllers = sorted(list(controllers))

        # 히트맵 데이터 생성
        heatmap_data = np.zeros((len(scenarios), len(controllers)))

        for i, scenario in enumerate(success_rates.keys()):
            scenario_title = scenario.replace('_', ' ').title()
            scenario_idx = scenarios.index(scenario_title)

            for j, controller in enumerate(controllers):
                if controller in success_rates[scenario]:
                    success_rate = success_rates[scenario][controller]['success_rate']
                    heatmap_data[scenario_idx, j] = success_rate * 100  # 백분율

        # 그래프 생성
        plt.figure(figsize=(10, 8))

        # 커스텀 컬러맵
        cmap = sns.diverging_palette(220, 20, as_cmap=True)

        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap=cmap,
                   xticklabels=controllers, yticklabels=scenarios,
                   cbar_kws={'label': 'Success Rate (%)'})

        plt.title('Controller Performance Heatmap by Scenario', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Controller', fontsize=12)
        plt.ylabel('Scenario', fontsize=12)
        plt.tight_layout()

        # 저장
        if save_path is None:
            save_path = self.output_dir / "scenario_performance_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 성능 히트맵 저장: {save_path}")
        return str(save_path)

    def extract_llm_output_examples(self, num_examples: int = 3) -> str:
        """LLM 출력 예시 추출"""
        if 'results' not in self.data:
            return "LLM 출력 데이터가 없습니다."

        examples = []

        # LLM-First 결과 찾기
        for scenario, scenario_results in self.data['results'].items():
            if 'LLM-First' in scenario_results and 'detailed_results' in scenario_results['LLM-First']:
                llm_results = scenario_results['LLM-First']['detailed_results']

                for result in llm_results[:num_examples]:
                    if 'controller_result' in result and 'params' in result['controller_result']:
                        params = result['controller_result']['params']
                        command = result.get('command', '')

                        example = f"""
### 시나리오: {scenario.replace('_', ' ').title()}
**명령어:** {command}

**물리 분석 결과:**
```json
{json.dumps(params.get('physical_analysis', {}), indent=2, ensure_ascii=False)}
```

**제어 파라미터:**
```json
{json.dumps(params.get('control_parameters', {}), indent=2, ensure_ascii=False)}
```

**추론 근거:** {params.get('reasoning', 'N/A')}

**평가 결과:** {'✅ 성공' if result.get('evaluation', {}).get('success', False) else '❌ 실패'}
---
"""
                        examples.append(example)

        if not examples:
            return "LLM 출력 예시를 찾을 수 없습니다."

        return "\n".join(examples)

    def generate_visualization_report(self, save_path: Optional[str] = None) -> str:
        """시각화 종합 보고서 생성"""
        report_lines = []
        report_lines.append("# 비교 실험 시각화 보고서")
        report_lines.append("")

        # 성공률 비교
        success_plot = self.create_success_rate_comparison()
        if success_plot:
            report_lines.append("## 1. 성공률 비교")
            report_lines.append(f"![성공률 비교]({Path(success_plot).name})")
            report_lines.append("")

        # 효율성 비교
        efficiency_plot = self.create_efficiency_comparison()
        if efficiency_plot:
            report_lines.append("## 2. 실행 효율성 비교")
            report_lines.append(f"![효율성 비교]({Path(efficiency_plot).name})")
            report_lines.append("")

        # 실패 분석
        failure_plot = self.create_failure_analysis_chart()
        if failure_plot:
            report_lines.append("## 3. 실패 유형 분석")
            report_lines.append(f"![실패 분석]({Path(failure_plot).name})")
            report_lines.append("")

        # 성능 히트맵
        heatmap_plot = self.create_scenario_performance_heatmap()
        if heatmap_plot:
            report_lines.append("## 4. 시나리오별 성능 히트맵")
            report_lines.append(f"![성능 히트맵]({Path(heatmap_plot).name})")
            report_lines.append("")

        # LLM 출력 예시
        report_lines.append("## 5. LLM 출력 예시")
        llm_examples = self.extract_llm_output_examples()
        report_lines.append(llm_examples)
        report_lines.append("")

        # 보고서 저장
        report_text = "\n".join(report_lines)
        if save_path is None:
            save_path = self.output_dir / "visualization_report.md"

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"✅ 시각화 보고서 저장: {save_path}")
        return str(save_path)

    def create_all_visualizations(self) -> Dict[str, str]:
        """모든 시각화 생성"""
        print("🎨 모든 시각화 생성 중...")

        visualizations = {}

        # 개별 그래프 생성
        visualizations['success_rate'] = self.create_success_rate_comparison()
        visualizations['efficiency'] = self.create_efficiency_comparison()
        visualizations['failure_analysis'] = self.create_failure_analysis_chart()
        visualizations['performance_heatmap'] = self.create_scenario_performance_heatmap()

        # 종합 보고서 생성
        visualizations['report'] = self.generate_visualization_report()

        print("✅ 모든 시각화 생성 완료")
        return visualizations


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='비교 실험 결과 시각화')
    parser.add_argument('results_file', help='실험 결과 JSON 파일 경로')
    parser.add_argument('--output-dir', help='시각화 저장 디렉토리 (기본: results_file과 같은 디렉토리)')

    args = parser.parse_args()

    if not os.path.exists(args.results_file):
        print(f"❌ 결과 파일을 찾을 수 없음: {args.results_file}")
        return

    # 시각화 생성
    visualizer = ComparisonVisualizer(args.results_file)
    visualizations = visualizer.create_all_visualizations()

    print("\n📊 생성된 시각화 파일들:")
    for name, path in visualizations.items():
        if path:
            print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
