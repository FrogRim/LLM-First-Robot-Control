#!/usr/bin/env python3
"""
성능 및 품질 메트릭 수집 (Performance & Quality Metrics Collection)

PhysicalAI → Genesis+Franka 파이프라인의 종합적인 성능 분석 및 품질 평가
"""

import sys
import time
import traceback
import logging
import psutil
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import gc
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# 프로젝트 모듈 임포트
sys.path.append('src')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceProfiler:
    """성능 프로파일러"""

    def __init__(self):
        self.metrics = {}
        self.start_time = None
        self.process = psutil.Process()

    def start_profiling(self, operation_name: str):
        """프로파일링 시작"""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.start_cpu = self.process.cpu_percent()

        if operation_name not in self.metrics:
            self.metrics[operation_name] = {
                'execution_times': [],
                'memory_usage': [],
                'cpu_usage': [],
                'errors': 0,
                'success': 0
            }

    def end_profiling(self, operation_name: str, success: bool = True):
        """프로파일링 종료 및 메트릭 기록"""
        if self.start_time is None:
            return

        execution_time = time.time() - self.start_time
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        end_cpu = self.process.cpu_percent()

        self.metrics[operation_name]['execution_times'].append(execution_time)
        self.metrics[operation_name]['memory_usage'].append(end_memory - self.start_memory)
        self.metrics[operation_name]['cpu_usage'].append(end_cpu)

        if success:
            self.metrics[operation_name]['success'] += 1
        else:
            self.metrics[operation_name]['errors'] += 1

        self.start_time = None

    def get_summary(self) -> Dict[str, Any]:
        """성능 요약 생성"""
        summary = {}

        for operation, data in self.metrics.items():
            if data['execution_times']:
                times = data['execution_times']
                memory = data['memory_usage']
                cpu = data['cpu_usage']

                summary[operation] = {
                    'avg_execution_time': np.mean(times),
                    'min_execution_time': np.min(times),
                    'max_execution_time': np.max(times),
                    'std_execution_time': np.std(times),
                    'avg_memory_delta': np.mean(memory),
                    'max_memory_delta': np.max(memory),
                    'avg_cpu_usage': np.mean(cpu),
                    'success_rate': data['success'] / (data['success'] + data['errors']) if (data['success'] + data['errors']) > 0 else 0,
                    'total_operations': data['success'] + data['errors']
                }

        return summary

def benchmark_data_processing():
    """데이터 처리 성능 벤치마크"""
    logger.info("=== 데이터 처리 성능 벤치마크 ===")

    profiler = PerformanceProfiler()

    try:
        from data_abstraction_layer import PhysicalAIAdapter, DataValidator

        # 샘플 파일 목록
        sample_files = list(Path("data/test_samples").glob("*.json"))
        valid_files = []

        # 유효한 파일만 필터링
        adapter = PhysicalAIAdapter({"debug": False})

        logger.info("유효한 샘플 파일 확인 중...")
        for file_path in sample_files:
            if adapter.validate_format(file_path):
                valid_files.append(file_path)

        logger.info(f"유효한 파일 수: {len(valid_files)}/{len(sample_files)}")

        if not valid_files:
            logger.error("유효한 샘플 파일이 없습니다.")
            return {}

        # 데이터 로딩 성능 테스트
        logger.info("데이터 로딩 성능 테스트...")
        for i, file_path in enumerate(valid_files):
            profiler.start_profiling('data_loading')

            try:
                schemas = list(adapter.load_dataset(file_path))
                success = len(schemas) > 0
                profiler.end_profiling('data_loading', success)

                if i % 1 == 0:  # 모든 파일에 대해 로그
                    logger.info(f"  파일 {i+1}/{len(valid_files)} 처리 완료 ({len(schemas)} 스키마)")

            except Exception as e:
                profiler.end_profiling('data_loading', False)
                logger.warning(f"  파일 {i+1} 로딩 실패: {e}")

        # 데이터 검증 성능 테스트
        logger.info("데이터 검증 성능 테스트...")
        validator = DataValidator({"debug": False})

        for file_path in valid_files[:3]:  # 처음 3개 파일만
            schemas = list(adapter.load_dataset(file_path))

            for j, schema in enumerate(schemas):
                profiler.start_profiling('data_validation')

                try:
                    result = validator.validate_schema(schema)
                    success = result['is_valid']
                    profiler.end_profiling('data_validation', success)

                except Exception as e:
                    profiler.end_profiling('data_validation', False)
                    logger.warning(f"  스키마 {j} 검증 실패: {e}")

        return profiler.get_summary()

    except Exception as e:
        logger.error(f"데이터 처리 벤치마크 실패: {e}")
        return {}

def benchmark_physics_mapping():
    """물리 매핑 성능 벤치마크"""
    logger.info("=== 물리 매핑 성능 벤치마크 ===")

    profiler = PerformanceProfiler()

    try:
        from physics_mapping_layer import UniversalPhysicsMapper, TrajectoryProcessor
        from data_abstraction_layer import PhysicalProperties, PhysicsEngine, TrajectoryData

        physics_mapper = UniversalPhysicsMapper({"debug": False})
        trajectory_processor = TrajectoryProcessor({"debug": False})

        # 다양한 물리 속성으로 테스트
        test_properties = [
            PhysicalProperties(mass=1.0, friction_coefficient=0.5, restitution=0.3, linear_damping=0.1, angular_damping=0.1, material_type="metal"),
            PhysicalProperties(mass=2.5, friction_coefficient=0.8, restitution=0.2, linear_damping=0.15, angular_damping=0.15, material_type="plastic"),
            PhysicalProperties(mass=0.5, friction_coefficient=0.3, restitution=0.8, linear_damping=0.05, angular_damping=0.05, material_type="rubber"),
            PhysicalProperties(mass=5.0, friction_coefficient=1.2, restitution=0.1, linear_damping=0.3, angular_damping=0.3, material_type="wood"),
        ]

        logger.info(f"물리 속성 변환 테스트 ({len(test_properties)}개 속성)...")

        for i, props in enumerate(test_properties * 5):  # 5회 반복
            profiler.start_profiling('physics_conversion')

            try:
                converted = physics_mapper.map_properties(
                    props,
                    PhysicsEngine.ISAAC_SIM,
                    PhysicsEngine.GENESIS_AI
                )

                # 변환 검증
                validation = physics_mapper.validate_conversion(props, converted)
                success = validation['is_valid']

                profiler.end_profiling('physics_conversion', success)

            except Exception as e:
                profiler.end_profiling('physics_conversion', False)
                logger.warning(f"  물리 변환 {i} 실패: {e}")

        # 궤적 처리 성능 테스트
        logger.info("궤적 처리 성능 테스트...")

        # 다양한 크기의 궤적 데이터 생성
        trajectory_sizes = [100, 500, 1000, 2000]

        for size in trajectory_sizes:
            timestamps = np.linspace(0, 10, size)  # 10초, 다양한 샘플링
            joint_positions = np.random.rand(size, 7)  # 7-DOF

            test_trajectory = TrajectoryData(
                timestamps=timestamps,
                joint_positions=joint_positions,
                sampling_rate=size/10.0
            )

            # 리샘플링 테스트
            for target_freq in [50, 100, 200]:
                profiler.start_profiling('trajectory_resampling')

                try:
                    resampled = trajectory_processor.resample_trajectory(
                        test_trajectory,
                        target_frequency=target_freq
                    )
                    success = len(resampled.timestamps) > 0
                    profiler.end_profiling('trajectory_resampling', success)

                except Exception as e:
                    profiler.end_profiling('trajectory_resampling', False)
                    logger.warning(f"  궤적 리샘플링 실패 (크기:{size}, 주파수:{target_freq}): {e}")

        return profiler.get_summary()

    except Exception as e:
        logger.error(f"물리 매핑 벤치마크 실패: {e}")
        return {}

def benchmark_language_generation():
    """언어 생성 성능 벤치마크"""
    logger.info("=== 언어 생성 성능 벤치마크 ===")

    profiler = PerformanceProfiler()

    try:
        from language_generation_layer import TemplateEngine, QualityAssuranceSystem, LanguageComplexity
        from data_abstraction_layer import PhysicalAIAdapter

        template_engine = TemplateEngine({"debug": False})
        qa_system = QualityAssuranceSystem({"debug": False})

        # 샘플 스키마 로드
        sample_files = list(Path("data/test_samples").glob("*.json"))
        adapter = PhysicalAIAdapter({"debug": False})

        valid_schemas = []
        for file_path in sample_files:
            if adapter.validate_format(file_path):
                schemas = list(adapter.load_dataset(file_path))
                valid_schemas.extend(schemas)

        if not valid_schemas:
            logger.error("유효한 스키마가 없습니다.")
            return {}

        logger.info(f"언어 생성 테스트 ({len(valid_schemas)} 스키마)...")

        # 각 복잡도 레벨로 테스트
        complexities = [LanguageComplexity.SIMPLE, LanguageComplexity.INTERMEDIATE, LanguageComplexity.ADVANCED]

        for schema in valid_schemas[:3]:  # 처음 3개 스키마만
            for complexity in complexities:
                # 명령 생성
                profiler.start_profiling('instruction_generation')
                try:
                    instruction = template_engine.generate_instruction(schema, complexity=complexity)
                    success = len(instruction.text) > 0
                    profiler.end_profiling('instruction_generation', success)
                except Exception as e:
                    profiler.end_profiling('instruction_generation', False)
                    logger.warning(f"  명령 생성 실패: {e}")

                # 설명 생성
                profiler.start_profiling('description_generation')
                try:
                    description = template_engine.generate_description(schema, complexity=complexity)
                    success = len(description.text) > 0
                    profiler.end_profiling('description_generation', success)
                except Exception as e:
                    profiler.end_profiling('description_generation', False)
                    logger.warning(f"  설명 생성 실패: {e}")

                # 물리 설명 생성
                profiler.start_profiling('explanation_generation')
                try:
                    explanation = template_engine.generate_explanation(
                        schema,
                        physics_focus=["mass", "friction"]
                    )
                    success = len(explanation.text) > 0
                    profiler.end_profiling('explanation_generation', success)
                except Exception as e:
                    profiler.end_profiling('explanation_generation', False)
                    logger.warning(f"  물리 설명 생성 실패: {e}")

        # 품질 평가 성능 테스트
        logger.info("품질 평가 성능 테스트...")

        # 테스트용 더미 어노테이션 생성
        if valid_schemas:
            schema = valid_schemas[0]

            for i in range(10):  # 10개 어노테이션 평가
                profiler.start_profiling('quality_assessment')

                try:
                    instruction = template_engine.generate_instruction(schema)
                    quality_metrics = qa_system.assess_quality(instruction)
                    validation = qa_system.validate_annotation(instruction)

                    success = quality_metrics.overall_score > 0
                    profiler.end_profiling('quality_assessment', success)

                except Exception as e:
                    profiler.end_profiling('quality_assessment', False)
                    logger.warning(f"  품질 평가 {i} 실패: {e}")

        return profiler.get_summary()

    except Exception as e:
        logger.error(f"언어 생성 벤치마크 실패: {e}")
        return {}

def benchmark_system_integration():
    """시스템 통합 성능 벤치마크"""
    logger.info("=== 시스템 통합 성능 벤치마크 ===")

    profiler = PerformanceProfiler()

    try:
        from config_monitoring_system import ConfigurationManager, MetricsCollector

        # 설정 관리 성능
        logger.info("설정 관리 성능 테스트...")

        config_manager = ConfigurationManager()

        # 설정 읽기 성능
        config_keys = [
            "data.batch_size", "physics.accuracy_threshold",
            "language.quality_threshold", "system.log_level"
        ]

        for i in range(100):  # 100회 설정 읽기
            profiler.start_profiling('config_read')

            try:
                for key in config_keys:
                    value = config_manager.get_config(key)
                profiler.end_profiling('config_read', True)

            except Exception as e:
                profiler.end_profiling('config_read', False)

        # 메트릭 수집 성능
        logger.info("메트릭 수집 성능 테스트...")

        metrics_collector = MetricsCollector(config_manager)

        for i in range(50):  # 50회 메트릭 기록
            profiler.start_profiling('metrics_collection')

            try:
                metrics_collector.record_metric(f"test.metric_{i%5}", np.random.random(), unit="test")
                metrics_collector.increment_counter("test.counter", 1)
                profiler.end_profiling('metrics_collection', True)

            except Exception as e:
                profiler.end_profiling('metrics_collection', False)

        # Genesis AI 초기화 성능
        logger.info("Genesis AI 초기화 성능 테스트...")

        for i in range(3):  # 3회 초기화 테스트
            profiler.start_profiling('genesis_initialization')

            try:
                import genesis
                genesis.init()
                scene = genesis.Scene()
                profiler.end_profiling('genesis_initialization', True)

            except Exception as e:
                profiler.end_profiling('genesis_initialization', False)
                logger.warning(f"  Genesis 초기화 {i} 실패: {e}")

        return profiler.get_summary()

    except Exception as e:
        logger.error(f"시스템 통합 벤치마크 실패: {e}")
        return {}

def generate_performance_report(all_metrics: Dict[str, Any]):
    """성능 보고서 생성"""
    logger.info("=== 성능 보고서 생성 ===")

    try:
        # 보고서 디렉토리 생성
        reports_dir = Path("reports/performance")
        reports_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON 보고서 생성
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "python_version": sys.version,
                "cpu_count": psutil.cpu_count(),
                "total_memory_gb": psutil.virtual_memory().total / 1024**3,
                "available_memory_gb": psutil.virtual_memory().available / 1024**3,
                "disk_usage_gb": psutil.disk_usage('/').total / 1024**3
            },
            "performance_metrics": all_metrics,
            "summary": generate_summary_metrics(all_metrics)
        }

        json_report_file = reports_dir / f"performance_report_{timestamp}.json"
        with open(json_report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        logger.info(f"✓ JSON 보고서 생성: {json_report_file}")

        # 텍스트 요약 보고서 생성
        text_report_file = reports_dir / f"performance_summary_{timestamp}.txt"
        with open(text_report_file, 'w', encoding='utf-8') as f:
            f.write("PhysicalAI → Genesis+Franka 파이프라인 성능 보고서\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 시스템 정보
            f.write("시스템 정보:\n")
            f.write(f"  CPU 코어 수: {psutil.cpu_count()}\n")
            f.write(f"  총 메모리: {psutil.virtual_memory().total / 1024**3:.1f} GB\n")
            f.write(f"  사용 가능 메모리: {psutil.virtual_memory().available / 1024**3:.1f} GB\n\n")

            # 성능 요약
            f.write("성능 요약:\n")
            summary = report_data["summary"]
            for category, metrics in summary.items():
                f.write(f"\n{category}:\n")
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"  {metric_name}: {value:.4f}\n")
                    else:
                        f.write(f"  {metric_name}: {value}\n")

        logger.info(f"✓ 텍스트 보고서 생성: {text_report_file}")

        # 시각화 차트 생성 (matplotlib이 있는 경우)
        try:
            create_performance_charts(all_metrics, reports_dir, timestamp)
        except Exception as e:
            logger.warning(f"차트 생성 실패: {e}")

        return json_report_file, text_report_file

    except Exception as e:
        logger.error(f"성능 보고서 생성 실패: {e}")
        return None, None

def generate_summary_metrics(all_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """종합 성능 메트릭 생성"""
    summary = {}

    for category, metrics in all_metrics.items():
        category_summary = {}

        # 평균 실행 시간
        avg_times = [m.get('avg_execution_time', 0) for m in metrics.values() if 'avg_execution_time' in m]
        if avg_times:
            category_summary['avg_execution_time'] = np.mean(avg_times)
            category_summary['total_operations'] = sum(m.get('total_operations', 0) for m in metrics.values())

        # 성공률
        success_rates = [m.get('success_rate', 0) for m in metrics.values() if 'success_rate' in m]
        if success_rates:
            category_summary['avg_success_rate'] = np.mean(success_rates)

        # 메모리 사용량
        memory_usage = [m.get('max_memory_delta', 0) for m in metrics.values() if 'max_memory_delta' in m]
        if memory_usage:
            category_summary['max_memory_usage_mb'] = max(memory_usage)

        if category_summary:
            summary[category] = category_summary

    return summary

def create_performance_charts(all_metrics: Dict[str, Any], output_dir: Path, timestamp: str):
    """성능 차트 생성"""
    try:
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('PhysicalAI → Genesis+Franka Pipeline Performance Analysis', fontsize=16)

        # 실행 시간 비교
        ax1 = axes[0, 0]
        categories = []
        avg_times = []

        for category, metrics in all_metrics.items():
            for operation, data in metrics.items():
                if 'avg_execution_time' in data:
                    categories.append(f"{category}\\n{operation}")
                    avg_times.append(data['avg_execution_time'] * 1000)  # ms 변환

        if categories and avg_times:
            ax1.bar(range(len(categories)), avg_times)
            ax1.set_xlabel('Operations')
            ax1.set_ylabel('Average Execution Time (ms)')
            ax1.set_title('Average Execution Time by Operation')
            ax1.set_xticks(range(len(categories)))
            ax1.set_xticklabels(categories, rotation=45, ha='right')

        # 성공률 비교
        ax2 = axes[0, 1]
        success_rates = []
        operation_names = []

        for category, metrics in all_metrics.items():
            for operation, data in metrics.items():
                if 'success_rate' in data:
                    operation_names.append(f"{category}.{operation}")
                    success_rates.append(data['success_rate'] * 100)

        if operation_names and success_rates:
            ax2.bar(range(len(operation_names)), success_rates)
            ax2.set_xlabel('Operations')
            ax2.set_ylabel('Success Rate (%)')
            ax2.set_title('Success Rate by Operation')
            ax2.set_ylim(0, 105)
            ax2.set_xticks(range(len(operation_names)))
            ax2.set_xticklabels(operation_names, rotation=45, ha='right')

        # 메모리 사용량
        ax3 = axes[1, 0]
        memory_usage = []
        memory_operations = []

        for category, metrics in all_metrics.items():
            for operation, data in metrics.items():
                if 'avg_memory_delta' in data:
                    memory_operations.append(f"{category}.{operation}")
                    memory_usage.append(abs(data['avg_memory_delta']))

        if memory_operations and memory_usage:
            ax3.bar(range(len(memory_operations)), memory_usage)
            ax3.set_xlabel('Operations')
            ax3.set_ylabel('Memory Usage (MB)')
            ax3.set_title('Memory Usage by Operation')
            ax3.set_xticks(range(len(memory_operations)))
            ax3.set_xticklabels(memory_operations, rotation=45, ha='right')

        # 전체 처리량 (operations per second)
        ax4 = axes[1, 1]
        throughput = []
        throughput_names = []

        for category, metrics in all_metrics.items():
            for operation, data in metrics.items():
                if 'avg_execution_time' in data and data['avg_execution_time'] > 0:
                    throughput_names.append(f"{category}.{operation}")
                    ops_per_sec = 1.0 / data['avg_execution_time']
                    throughput.append(ops_per_sec)

        if throughput_names and throughput:
            ax4.bar(range(len(throughput_names)), throughput)
            ax4.set_xlabel('Operations')
            ax4.set_ylabel('Operations per Second')
            ax4.set_title('Throughput by Operation')
            ax4.set_xticks(range(len(throughput_names)))
            ax4.set_xticklabels(throughput_names, rotation=45, ha='right')

        plt.tight_layout()

        chart_file = output_dir / f"performance_charts_{timestamp}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ 성능 차트 생성: {chart_file}")

    except Exception as e:
        logger.warning(f"차트 생성 중 오류: {e}")

def main():
    """메인 성능 메트릭 수집 실행"""
    logger.info("성능 및 품질 메트릭 수집 시작")
    logger.info("=" * 60)

    start_time = time.time()

    # 각 모듈별 성능 벤치마크 실행
    all_metrics = {}

    logger.info("1. 데이터 처리 성능 벤치마크 실행...")
    all_metrics['data_processing'] = benchmark_data_processing()

    logger.info("2. 물리 매핑 성능 벤치마크 실행...")
    all_metrics['physics_mapping'] = benchmark_physics_mapping()

    logger.info("3. 언어 생성 성능 벤치마크 실행...")
    all_metrics['language_generation'] = benchmark_language_generation()

    logger.info("4. 시스템 통합 성능 벤치마크 실행...")
    all_metrics['system_integration'] = benchmark_system_integration()

    total_time = time.time() - start_time

    # 성능 보고서 생성
    logger.info("5. 성능 보고서 생성...")
    json_report, text_report = generate_performance_report(all_metrics)

    # 결과 요약
    logger.info("\n" + "=" * 60)
    logger.info("성능 메트릭 수집 완료!")
    logger.info(f"총 소요 시간: {total_time:.2f}초")

    if json_report and text_report:
        logger.info(f"JSON 보고서: {json_report}")
        logger.info(f"텍스트 보고서: {text_report}")

    # 종합 성능 점수 계산
    summary = generate_summary_metrics(all_metrics)

    logger.info("\n종합 성능 요약:")
    for category, metrics in summary.items():
        logger.info(f"  {category}:")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"    {metric_name}: {value:.4f}")
            else:
                logger.info(f"    {metric_name}: {value}")

    logger.info("\n🎯 PhysicalAI → Genesis+Franka 파이프라인 성능 분석 완료!")

    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("성능 메트릭 수집이 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"성능 메트릭 수집 중 치명적 오류: {e}")
        traceback.print_exc()
        sys.exit(1)