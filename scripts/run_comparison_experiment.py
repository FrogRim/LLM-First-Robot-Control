#!/usr/bin/env python3
"""
비교 실험 메인 실행 스크립트
Rule-Based, RL 기반, LLM-First 제어 시스템을 3가지 시나리오에서 비교 평가
"""

import json
import time
import argparse
import os
from typing import Dict, Any, List
from datetime import datetime

# 컨트롤러 및 시나리오 임포트
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rule_based_controller import RuleBasedController
from rl_agent_controller import RLController
from llm_first_controller import LLMFirstController
from experiment_scenarios import create_scenario, run_scenario_test
from evaluation_metrics import ExperimentEvaluator


def run_full_comparison_experiment(num_runs_per_scenario: int = 10,
                                 enable_genesis: bool = True,
                                 output_dir: str = "./comparison_results") -> Dict[str, Any]:
    """
    전체 비교 실험 실행

    Args:
        num_runs_per_scenario: 각 시나리오당 실행 횟수
        enable_genesis: Genesis 시뮬레이션 활성화
        output_dir: 결과 저장 디렉토리

    Returns:
        실험 결과 요약
    """
    print("🚀 LLM-First vs 기존 방식 비교 실험 시작")
    print("="*80)
    print(f"실행 설정: 각 시나리오당 {num_runs_per_scenario}회 반복")
    print(f"Genesis 시뮬레이션: {'활성화' if enable_genesis else '비활성화'}")
    print(f"결과 저장 위치: {output_dir}")
    print("="*80)

    # 결과 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 컨트롤러 설정
    controllers = {
        'Rule-Based': RuleBasedController,
        'RL-Based': RLController,
        'LLM-First': LLMFirstController
    }

    # 시나리오 설정
    scenario_types = [
        'object_sorting',
        'multi_step_rearrangement',
        'physical_property_reasoning'
    ]

    # 평가자 초기화
    evaluator = ExperimentEvaluator()

    # 전체 결과 저장
    all_results = []
    experiment_summary = {
        'experiment_config': {
            'num_runs_per_scenario': num_runs_per_scenario,
            'enable_genesis': enable_genesis,
            'timestamp': datetime.now().isoformat(),
            'controllers': list(controllers.keys()),
            'scenarios': scenario_types
        },
        'results': {},
        'performance_comparison': {}
    }

    total_start_time = time.time()

    try:
        # 각 시나리오별로 실험 실행
        for scenario_type in scenario_types:
            print(f"\n🎯 시나리오: {scenario_type.upper()}")
            print("-" * 60)

            scenario_results = {}

            # 시나리오별로 하나의 씬 생성 (컨트롤러들이 공유)
            print("🌍 시나리오용 Genesis 씬 생성 중...")
            shared_scenario = create_scenario(scenario_type, enable_genesis)

            # 각 컨트롤러별로 테스트 (같은 씬 공유)
            for controller_name, controller_class in controllers.items():
                print(f"\n🤖 컨트롤러: {controller_name}")

                try:
                    # 시나리오 테스트 실행 (공유된 시나리오 사용)
                    start_time = time.time()
                    summary = run_scenario_test(
                        shared_scenario, controller_class, controller_name,
                        num_runs=num_runs_per_scenario
                    )
                    execution_time = time.time() - start_time

                    # 결과 저장
                    summary['execution_time_sec'] = execution_time
                    scenario_results[controller_name] = summary

                    # 평가자에 결과 추가
                    for result in summary['detailed_results']:
                        evaluator.add_result(scenario_type, controller_name, result)
                        all_results.append({
                            'scenario_type': scenario_type,
                            'controller_name': controller_name,
                            **result
                        })

                    print(f"✅ {controller_name} 완료: {execution_time:.1f}초")
                except Exception as e:
                    print(f"❌ {controller_name} 실행 실패: {e}")
                    scenario_results[controller_name] = {
                        'error': str(e),
                        'success': False
                    }

                # 컨트롤러 완료 후 씬 리셋 (다음 컨트롤러를 위해)
                if hasattr(shared_scenario, 'reset_scene'):
                    shared_scenario.reset_scene()

            # 시나리오별 메모리 정리
            if hasattr(shared_scenario, 'cleanup'):
                shared_scenario.cleanup()

            experiment_summary['results'][scenario_type] = scenario_results

        # 전체 실행 시간
        total_execution_time = time.time() - total_start_time
        experiment_summary['experiment_config']['total_execution_time_sec'] = total_execution_time

        print(f"\n⏱️  총 실행 시간: {total_execution_time:.1f}초")
        print("\n📊 성능 비교 분석 중...")

        # 성능 비교 분석
        experiment_summary['performance_comparison'] = {
            'success_rates': {},
            'efficiency_metrics': {},
            'failure_analysis': {},
            'controller_comparison': {}
        }

        # 성공률 분석
        for scenario in scenario_types:
            comparison = evaluator.compare_controllers(scenario)
            experiment_summary['performance_comparison']['success_rates'][scenario] = comparison

        # 효율성 메트릭
        for controller in controllers.keys():
            efficiency = evaluator.calculate_efficiency_metrics(controller_filter=controller)
            experiment_summary['performance_comparison']['efficiency_metrics'][controller] = efficiency

            failure_analysis = evaluator.analyze_failure_patterns(controller_filter=controller)
            experiment_summary['performance_comparison']['failure_analysis'][controller] = failure_analysis

        # 컨트롤러별 종합 비교
        experiment_summary['performance_comparison']['controller_comparison'] = evaluator.compare_controllers(None)

        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON 결과 저장
        json_file = f"{output_dir}/comparison_results_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_summary, f, indent=2, ensure_ascii=False, default=str)
        print(f"✅ JSON 결과 저장: {json_file}")

        # 상세 결과 저장
        detailed_file = f"{output_dir}/detailed_results_{timestamp}.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"✅ 상세 결과 저장: {detailed_file}")

        # 평가 보고서 생성
        report_file = f"{output_dir}/evaluation_report_{timestamp}.md"
        report = evaluator.generate_report(report_file)
        print(f"✅ 평가 보고서 저장: {report_file}")

        print("\n🎉 비교 실험 완료!")
        print("="*80)

        return experiment_summary

    except Exception as e:
        print(f"❌ 실험 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

        # 에러 정보 저장
        error_summary = {
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'partial_results': experiment_summary
        }

        error_file = f"{output_dir}/experiment_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(error_summary, f, indent=2, ensure_ascii=False, default=str)

        print(f"에러 정보 저장: {error_file}")
        return error_summary


def run_quick_test(num_runs: int = 3) -> Dict[str, Any]:
    """빠른 테스트 실행 (디버깅용)"""
    print("🧪 빠른 비교 실험 테스트")
    print(f"각 시나리오당 {num_runs}회 실행")

    return run_full_comparison_experiment(
        num_runs_per_scenario=num_runs,
        enable_genesis=True,
        output_dir="./quick_test_results"
    )


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='LLM-First vs 기존 방식 비교 실험')
    parser.add_argument('--runs', type=int, default=10,
                       help='각 시나리오당 실행 횟수 (기본: 10)')
    parser.add_argument('--no-genesis', action='store_true',
                       help='Genesis 시뮬레이션 비활성화')
    parser.add_argument('--output-dir', type=str, default='./comparison_results',
                       help='결과 저장 디렉토리')
    parser.add_argument('--quick-test', action='store_true',
                       help='빠른 테스트 모드 (3회 실행)')

    args = parser.parse_args()

    if args.quick_test:
        run_quick_test(num_runs=3)
    else:
        run_full_comparison_experiment(
            num_runs_per_scenario=args.runs,
            enable_genesis=not args.no_genesis,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
