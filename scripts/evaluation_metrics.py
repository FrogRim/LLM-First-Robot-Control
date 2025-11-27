#!/usr/bin/env python3
"""
비교 실험 평가 지표 측정 및 분석
성공률, 계획 일관성, 실행 효율성, 실패 유형 분석
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from collections import defaultdict, Counter
import statistics
from scipy import stats


class ExperimentEvaluator:
    """실험 결과 평가자 클래스"""

    def __init__(self):
        self.results_data = []
        self.failure_analysis = defaultdict(list)

    def add_result(self, scenario_type: str, controller_name: str, result: Dict[str, Any]):
        """실험 결과 추가"""
        evaluation = result.get('evaluation', {})
        controller_result = result.get('controller_result', {})

        data_point = {
            'scenario_type': scenario_type,
            'controller_name': controller_name,
            'run_idx': result.get('run_idx', 0),
            'command': result.get('command', ''),
            'success': evaluation.get('success', False),
            'total_time_sec': result.get('total_time_sec', 0),
            'inference_time_ms': evaluation.get('inference_time_ms', 0),
            'has_plan': evaluation.get('has_plan', False),
            'material_correct': evaluation.get('material_correct', False),
            'parameter_appropriate': evaluation.get('parameter_appropriate', True),
            'detected_material': evaluation.get('detected_material', ''),
            'expected_materials': evaluation.get('expected_materials', []),
            'control_parameters': evaluation.get('control_parameters', {}),
            'reasoning': evaluation.get('reasoning', ''),
            'raw_controller_result': controller_result
        }

        self.results_data.append(data_point)

        # 실패 분석용 데이터 수집
        if not evaluation.get('success', False):
            failure_reason = evaluation.get('reasoning', 'Unknown failure')
            self.failure_analysis[controller_name].append({
                'scenario': scenario_type,
                'command': result.get('command', ''),
                'reason': failure_reason,
                'inference_time': evaluation.get('inference_time_ms', 0)
            })

    def calculate_success_rate(self, scenario_filter: Optional[str] = None,
                              controller_filter: Optional[str] = None) -> Dict[str, Any]:
        """성공률 계산"""
        filtered_data = self._filter_data(scenario_filter, controller_filter)

        if not filtered_data:
            return {'success_rate': 0.0, 'total_runs': 0, 'successful_runs': 0}

        successful_runs = sum(1 for d in filtered_data if d['success'])
        total_runs = len(filtered_data)
        success_rate = successful_runs / total_runs if total_runs > 0 else 0

        return {
            'success_rate': success_rate,
            'total_runs': total_runs,
            'successful_runs': successful_runs,
            'confidence_interval': self._calculate_confidence_interval(successful_runs, total_runs)
        }

    def calculate_efficiency_metrics(self, scenario_filter: Optional[str] = None,
                                   controller_filter: Optional[str] = None) -> Dict[str, Any]:
        """실행 효율성 메트릭 계산"""
        filtered_data = self._filter_data(scenario_filter, controller_filter)

        if not filtered_data:
            return {}

        inference_times = [d['inference_time_ms'] for d in filtered_data if d['inference_time_ms'] > 0]
        total_times = [d['total_time_sec'] for d in filtered_data if d['total_time_sec'] > 0]

        metrics = {}

        if inference_times:
            metrics['inference_time'] = {
                'mean': statistics.mean(inference_times),
                'median': statistics.median(inference_times),
                'std': statistics.stdev(inference_times) if len(inference_times) > 1 else 0,
                'min': min(inference_times),
                'max': max(inference_times)
            }

        if total_times:
            metrics['total_time'] = {
                'mean': statistics.mean(total_times),
                'median': statistics.median(total_times),
                'std': statistics.stdev(total_times) if len(total_times) > 1 else 0,
                'min': min(total_times),
                'max': max(total_times)
            }

        return metrics

    def analyze_plan_consistency(self, scenario_filter: Optional[str] = None,
                               controller_filter: Optional[str] = None) -> Dict[str, Any]:
        """계획 일관성 분석"""
        filtered_data = self._filter_data(scenario_filter, controller_filter)

        if not filtered_data:
            return {}

        plan_indicators = [d['has_plan'] for d in filtered_data]
        plan_consistency = sum(plan_indicators) / len(plan_indicators) if plan_indicators else 0

        # 계획 일관성과 성공률의 상관관계
        successful_with_plan = sum(1 for d in filtered_data if d['success'] and d['has_plan'])
        successful_without_plan = sum(1 for d in filtered_data if d['success'] and not d['has_plan'])

        return {
            'plan_consistency_rate': plan_consistency,
            'successful_with_plan': successful_with_plan,
            'successful_without_plan': successful_without_plan,
            'correlation_plan_success': self._calculate_correlation(plan_indicators,
                                                                   [d['success'] for d in filtered_data])
        }

    def analyze_failure_patterns(self, controller_filter: Optional[str] = None) -> Dict[str, Any]:
        """실패 유형 분석"""
        if controller_filter:
            failures = self.failure_analysis.get(controller_filter, [])
        else:
            failures = []
            for controller_failures in self.failure_analysis.values():
                failures.extend(controller_failures)

        if not failures:
            return {'total_failures': 0, 'patterns': {}}

        # 실패 이유 카테고리화
        failure_categories = {
            'material_detection': 0,
            'parameter_inappropriate': 0,
            'planning_issue': 0,
            'execution_error': 0,
            'unknown': 0
        }

        for failure in failures:
            reason = failure['reason'].lower()
            if 'material' in reason or 'detect' in reason:
                failure_categories['material_detection'] += 1
            elif 'parameter' in reason or 'appropriate' in reason:
                failure_categories['parameter_inappropriate'] += 1
            elif 'plan' in reason or 'sequence' in reason:
                failure_categories['planning_issue'] += 1
            elif 'execution' in reason or 'error' in reason:
                failure_categories['execution_error'] += 1
            else:
                failure_categories['unknown'] += 1

        # 시나리오별 실패 분석
        scenario_failures = Counter(f['scenario'] for f in failures)

        return {
            'total_failures': len(failures),
            'failure_categories': failure_categories,
            'scenario_distribution': dict(scenario_failures),
            'average_inference_time_on_failure': statistics.mean([f['inference_time'] for f in failures]) if failures else 0
        }

    def compare_controllers(self, scenario_type: str) -> Dict[str, Any]:
        """컨트롤러 간 비교 분석"""
        controllers = set(d['controller_name'] for d in self.results_data
                         if d['scenario_type'] == scenario_type)

        comparison = {}
        for controller in controllers:
            controller_data = [d for d in self.results_data
                             if d['scenario_type'] == scenario_type and d['controller_name'] == controller]

            if controller_data:
                success_rate = sum(1 for d in controller_data if d['success']) / len(controller_data)
                avg_inference = statistics.mean([d['inference_time_ms'] for d in controller_data if d['inference_time_ms'] > 0])
                avg_total_time = statistics.mean([d['total_time_sec'] for d in controller_data if d['total_time_sec'] > 0])

                comparison[controller] = {
                    'success_rate': success_rate,
                    'avg_inference_time_ms': avg_inference,
                    'avg_total_time_sec': avg_total_time,
                    'runs': len(controller_data)
                }

        return comparison

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """종합 평가 보고서 생성"""
        report_lines = []
        report_lines.append("# 비교 실험 평가 보고서")
        report_lines.append("")

        # 전체 통계
        total_runs = len(self.results_data)
        total_success = sum(1 for d in self.results_data if d['success'])
        overall_success_rate = total_success / total_runs if total_runs > 0 else 0

        report_lines.append("## 전체 통계")
        report_lines.append(f"- 총 실험 수: {total_runs}")
        report_lines.append(f"- 전체 성공률: {overall_success_rate:.1%}")
        report_lines.append(f"- 평균 추론 시간: {self._safe_mean([d['inference_time_ms'] for d in self.results_data if d['inference_time_ms'] > 0]):.1f}ms")
        report_lines.append("")

        # 시나리오별 분석
        scenarios = set(d['scenario_type'] for d in self.results_data)
        report_lines.append("## 시나리오별 분석")

        for scenario in scenarios:
            scenario_data = [d for d in self.results_data if d['scenario_type'] == scenario]
            scenario_success = sum(1 for d in scenario_data if d['success'])
            scenario_rate = scenario_success / len(scenario_data) if scenario_data else 0

            report_lines.append(f"### {scenario}")
            report_lines.append(f"- 실행 수: {len(scenario_data)}")
            report_lines.append(f"- 성공률: {scenario_rate:.1%}")

            # 컨트롤러별 비교
            comparison = self.compare_controllers(scenario)
            if comparison:
                report_lines.append("- 컨트롤러별 성능:")
                for controller, metrics in comparison.items():
                    report_lines.append(f"  - {controller}: {metrics['success_rate']:.1%} 성공률, {metrics['avg_inference_time_ms']:.1f}ms 추론시간")
            report_lines.append("")

        # 컨트롤러별 분석
        controllers = set(d['controller_name'] for d in self.results_data)
        report_lines.append("## 컨트롤러별 분석")

        for controller in controllers:
            controller_data = [d for d in self.results_data if d['controller_name'] == controller]
            controller_success = sum(1 for d in controller_data if d['success'])
            controller_rate = controller_success / len(controller_data) if controller_data else 0

            report_lines.append(f"### {controller}")
            report_lines.append(f"- 실행 수: {len(controller_data)}")
            report_lines.append(f"- 성공률: {controller_rate:.1%}")

            # 실패 분석
            failure_analysis = self.analyze_failure_patterns(controller)
            if failure_analysis['total_failures'] > 0:
                report_lines.append("- 실패 분석:")
                categories = failure_analysis['failure_categories']
                for category, count in categories.items():
                    if count > 0:
                        report_lines.append(f"  - {category}: {count}건")
            report_lines.append("")

        # 보고서 저장
        report_text = "\n".join(report_lines)
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"보고서 저장됨: {output_file}")

        return report_text

    def _filter_data(self, scenario_filter: Optional[str] = None,
                    controller_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """데이터 필터링"""
        filtered = self.results_data

        if scenario_filter:
            filtered = [d for d in filtered if d['scenario_type'] == scenario_filter]

        if controller_filter:
            filtered = [d for d in filtered if d['controller_name'] == controller_filter]

        return filtered

    def _calculate_confidence_interval(self, successes: int, total: int, confidence: float = 0.95) -> tuple:
        """성공률에 대한 신뢰구간 계산"""
        if total == 0:
            return (0.0, 0.0)

        p_hat = successes / total
        z = stats.norm.ppf((1 + confidence) / 2)
        margin = z * np.sqrt(p_hat * (1 - p_hat) / total)

        return (max(0, p_hat - margin), min(1, p_hat + margin))

    def _calculate_correlation(self, x: List[bool], y: List[bool]) -> float:
        """두 이진 변수 간 상관계수 계산 (점 양분 상관계수)"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        x_numeric = [1 if val else 0 for val in x]
        y_numeric = [1 if val else 0 for val in y]

        try:
            correlation = np.corrcoef(x_numeric, y_numeric)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0

    def _safe_mean(self, values: List[float]) -> float:
        """안전한 평균 계산"""
        return statistics.mean(values) if values else 0.0


def load_experiment_results(results_file: str) -> ExperimentEvaluator:
    """저장된 실험 결과 로드"""
    evaluator = ExperimentEvaluator()

    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

        for result in results:
            evaluator.add_result(
                result['scenario_type'],
                result['controller_name'],
                result
            )

        print(f"실험 결과 로드 완료: {len(results)}개 결과")
        return evaluator

    except FileNotFoundError:
        print(f"결과 파일을 찾을 수 없음: {results_file}")
        return evaluator
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        return evaluator


if __name__ == "__main__":
    """평가 메트릭 테스트"""
    print("🧪 평가 메트릭 테스트")

    # 샘플 데이터로 테스트
    evaluator = ExperimentEvaluator()

    # 샘플 결과 추가
    sample_results = [
        {
            'scenario_type': 'object_sorting',
            'controller_name': 'Rule-Based',
            'run_idx': 0,
            'command': 'Sort plastic bottle to left',
            'total_time_sec': 1.2,
            'evaluation': {
                'success': True,
                'inference_time_ms': 0.5,
                'material_correct': True,
                'reasoning': 'Correctly identified plastic material'
            },
            'controller_result': {}
        },
        {
            'scenario_type': 'object_sorting',
            'controller_name': 'LLM-First',
            'run_idx': 0,
            'command': 'Sort plastic bottle to left',
            'total_time_sec': 15.8,
            'evaluation': {
                'success': True,
                'inference_time_ms': 15600,
                'material_correct': True,
                'reasoning': 'LLM correctly inferred plastic properties'
            },
            'controller_result': {}
        }
    ]

    for result in sample_results:
        evaluator.add_result(
            result['scenario_type'],
            result['controller_name'],
            result
        )

    # 보고서 생성
    report = evaluator.generate_report()
    print("\n" + "="*50)
    print("샘플 평가 보고서:")
    print("="*50)
    print(report)
