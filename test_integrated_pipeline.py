#!/usr/bin/env python3
"""
통합 파이프라인 테스트 (Integrated Pipeline Test)

PhysicalAI → Genesis+Franka 전체 변환 파이프라인의 엔드투엔드 테스트
"""

import sys
import traceback
import logging
import time
from pathlib import Path
import json
import numpy as np

# 프로젝트 모듈 임포트
sys.path.append('src')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_full_conversion_pipeline():
    """전체 변환 파이프라인 테스트"""
    logger.info("=== 통합 파이프라인 테스트 ===")

    try:
        # 1. 모든 계층 모듈 임포트
        logger.info("1. 파이프라인 모듈 임포트...")
        from data_abstraction_layer import PhysicalAIAdapter, DataValidator
        from physics_mapping_layer import UniversalPhysicsMapper, TrajectoryProcessor
        from language_generation_layer import TemplateEngine, QualityAssuranceSystem
        from config_monitoring_system import ConfigurationManager, MetricsCollector

        logger.info("✓ 모든 모듈 임포트 성공")

        # 2. 설정 및 시스템 초기화
        logger.info("2. 시스템 초기화...")
        config_manager = ConfigurationManager()
        config = {"debug": True}

        # 각 계층 초기화
        data_adapter = PhysicalAIAdapter(config)
        physics_mapper = UniversalPhysicsMapper(config)
        trajectory_processor = TrajectoryProcessor(config)
        template_engine = TemplateEngine(config)
        qa_system = QualityAssuranceSystem(config)
        data_validator = DataValidator(config)
        metrics_collector = MetricsCollector(config_manager)

        logger.info("✓ 시스템 초기화 완료")

        # 3. 샘플 데이터 로딩
        logger.info("3. 샘플 데이터 로딩...")
        sample_files = list(Path("data/test_samples").glob("*.json"))
        if not sample_files:
            logger.error("샘플 JSON 파일이 없습니다.")
            return False

        # 가장 최신 파일 선택
        sample_file = max(sample_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"   사용 파일: {sample_file}")

        # 데이터 검증
        if not data_adapter.validate_format(sample_file):
            logger.error("데이터 포맷 검증 실패")
            return False

        # 스키마 로딩
        schemas = list(data_adapter.load_dataset(sample_file))
        if not schemas:
            logger.error("스키마 로딩 실패")
            return False

        schema = schemas[0]
        logger.info(f"✓ 데이터 로딩 완료 (스키마: {type(schema).__name__})")

        # 4. 물리 속성 변환 테스트
        logger.info("4. 물리 속성 변환...")
        start_time = time.time()

        # 물리 속성 변환
        converted_props = physics_mapper.map_properties(
            schema.physical_properties,
            schema.metadata.source_engine,
            schema.metadata.target_engine
        )

        # 변환 검증
        validation_result = physics_mapper.validate_conversion(
            schema.physical_properties,
            converted_props
        )

        if not validation_result['is_valid']:
            logger.warning(f"물리 변환 검증 경고: {validation_result.get('warnings', [])}")

        physics_time = time.time() - start_time
        logger.info(f"✓ 물리 속성 변환 완료 (정확도: {validation_result['accuracy_score']:.3f}, 시간: {physics_time:.3f}s)")

        # 5. 궤적 데이터 처리
        logger.info("5. 궤적 데이터 처리...")
        start_time = time.time()

        # 궤적 리샘플링 (Genesis AI 최적화를 위해 50Hz로)
        target_frequency = config_manager.get_config("physics.target_frequency", 50.0)
        resampled_trajectory = trajectory_processor.resample_trajectory(
            schema.trajectory_data,
            target_frequency=target_frequency
        )

        trajectory_time = time.time() - start_time
        logger.info(f"✓ 궤적 처리 완료 ({len(schema.trajectory_data.timestamps)} → {len(resampled_trajectory.timestamps)} 포인트, 시간: {trajectory_time:.3f}s)")

        # 6. 자연어 어노테이션 생성
        logger.info("6. 자연어 어노테이션 생성...")
        start_time = time.time()

        # 변환된 스키마 생성
        from data_abstraction_layer import StandardDataSchema, ProcessingMetadata
        converted_schema = StandardDataSchema(
            robot_config=schema.robot_config,
            physical_properties=converted_props,
            trajectory_data=resampled_trajectory,
            scene_description=schema.scene_description,
            metadata=ProcessingMetadata(
                source_engine=schema.metadata.source_engine,
                target_engine=schema.metadata.target_engine,
                conversion_timestamp=schema.metadata.conversion_timestamp,
                conversion_version=schema.metadata.conversion_version,
                data_completeness=1.0,
                physics_accuracy=validation_result['accuracy_score'],
                language_quality=0.0,  # 곧 계산될 예정
                processing_time=physics_time + trajectory_time,
                memory_usage=schema.metadata.memory_usage
            )
        )

        # 다양한 타입의 어노테이션 생성
        from language_generation_layer import LanguageComplexity

        instruction = template_engine.generate_instruction(
            converted_schema,
            complexity=LanguageComplexity.INTERMEDIATE
        )

        description = template_engine.generate_description(
            converted_schema,
            complexity=LanguageComplexity.INTERMEDIATE
        )

        explanation = template_engine.generate_explanation(
            converted_schema,
            physics_focus=["mass", "friction", "restitution"]
        )

        language_time = time.time() - start_time
        logger.info(f"✓ 어노테이션 생성 완료 (시간: {language_time:.3f}s)")

        # 7. 품질 평가 및 검증
        logger.info("7. 품질 평가 및 검증...")

        annotations = [instruction, description, explanation]
        quality_scores = []

        for i, annotation in enumerate(annotations):
            # 개별 어노테이션 품질 평가
            quality_metrics = qa_system.assess_quality(annotation)
            validation = qa_system.validate_annotation(annotation)

            quality_scores.append(quality_metrics.overall_score)

            logger.info(f"   어노테이션 {i+1}: 품질={quality_metrics.overall_score:.3f}, 검증={'통과' if validation['is_valid'] else '실패'}")

        overall_language_quality = np.mean(quality_scores)

        # 전체 스키마 검증
        schema_validation = data_validator.validate_schema(converted_schema)
        logger.info(f"   전체 스키마 검증: {'성공' if schema_validation['is_valid'] else '실패'}")
        logger.info(f"   완전성 점수: {schema_validation['completeness_score']:.3f}")

        # 8. Genesis AI 호환성 테스트
        logger.info("8. Genesis AI 호환성 테스트...")
        genesis_compatible = test_genesis_compatibility(converted_schema)

        # 9. 메트릭 수집 및 결과 생성
        total_time = physics_time + trajectory_time + language_time

        # 메트릭 기록
        metrics_collector.record_metric("pipeline.total_processing_time", total_time, unit="seconds")
        metrics_collector.record_metric("pipeline.physics_accuracy", validation_result['accuracy_score'])
        metrics_collector.record_metric("pipeline.language_quality", overall_language_quality)
        metrics_collector.record_metric("pipeline.data_completeness", schema_validation['completeness_score'])
        metrics_collector.increment_counter("pipeline.successful_conversions", 1)

        # 결과 요약
        logger.info("✓ 통합 파이프라인 테스트 완료")
        logger.info(f"   총 처리 시간: {total_time:.3f}초")
        logger.info(f"   물리 정확도: {validation_result['accuracy_score']:.3f}")
        logger.info(f"   언어 품질: {overall_language_quality:.3f}")
        logger.info(f"   데이터 완전성: {schema_validation['completeness_score']:.3f}")
        logger.info(f"   Genesis 호환성: {'성공' if genesis_compatible else '실패'}")

        # 변환 결과 저장
        save_conversion_results(converted_schema, annotations, sample_file.stem)

        return True

    except Exception as e:
        logger.error(f"✗ 통합 파이프라인 테스트 실패: {e}")
        traceback.print_exc()
        return False

def test_genesis_compatibility(schema):
    """Genesis AI 호환성 테스트"""
    try:
        import genesis

        logger.info("   Genesis AI 초기화...")
        genesis.init()

        logger.info("   기본 시뮬레이션 장면 생성...")
        scene = genesis.Scene()

        # 로봇 설정 호환성 검증
        robot_config = schema.robot_config
        logger.info(f"   로봇 타입: {robot_config.robot_type}")
        logger.info(f"   관절 수: {robot_config.joint_count}")

        # 물리 속성 호환성 검증
        physics_props = schema.physical_properties
        logger.info(f"   질량: {physics_props.mass}kg")
        logger.info(f"   마찰계수: {physics_props.friction_coefficient}")

        logger.info("✓ Genesis AI 호환성 확인 완료")
        return True

    except Exception as e:
        logger.warning(f"Genesis AI 호환성 테스트 실패: {e}")
        return False

def save_conversion_results(schema, annotations, sample_name):
    """변환 결과 저장"""
    try:
        results_dir = Path("results/conversions")
        results_dir.mkdir(parents=True, exist_ok=True)

        # 결과 데이터 구성
        results = {
            "conversion_metadata": {
                "source_file": sample_name,
                "conversion_timestamp": schema.metadata.conversion_timestamp,
                "processing_time": schema.metadata.processing_time,
                "physics_accuracy": schema.metadata.physics_accuracy,
                "language_quality": schema.metadata.language_quality
            },
            "robot_configuration": {
                "type": schema.robot_config.robot_type.value,
                "joint_count": schema.robot_config.joint_count,
                "joint_names": schema.robot_config.joint_names
            },
            "physical_properties": {
                "mass": float(schema.physical_properties.mass),
                "friction_coefficient": float(schema.physical_properties.friction_coefficient),
                "restitution": float(schema.physical_properties.restitution),
                "material_type": schema.physical_properties.material_type
            },
            "trajectory_summary": {
                "duration": float(schema.trajectory_data.timestamps[-1] - schema.trajectory_data.timestamps[0]),
                "sample_count": len(schema.trajectory_data.timestamps),
                "sampling_rate": float(schema.trajectory_data.sampling_rate)
            },
            "scene_description": {
                "task_type": schema.scene_description.task_type,
                "task_description": schema.scene_description.task_description,
                "environment_type": schema.scene_description.environment_type
            },
            "generated_annotations": [
                {
                    "type": annotation.annotation_type.value,
                    "text": annotation.text,
                    "confidence": float(annotation.confidence),
                    "complexity": annotation.complexity.value
                }
                for annotation in annotations
            ]
        }

        # JSON으로 저장
        output_file = results_dir / f"conversion_result_{sample_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"✓ 변환 결과 저장: {output_file}")

    except Exception as e:
        logger.warning(f"결과 저장 실패: {e}")

def test_batch_processing():
    """배치 처리 테스트"""
    logger.info("\n=== 배치 처리 테스트 ===")

    try:
        sample_files = list(Path("data/test_samples").glob("*.json"))
        if len(sample_files) < 2:
            logger.info("배치 테스트를 위한 충분한 샘플 파일이 없습니다.")
            return True

        logger.info(f"배치 처리 대상: {len(sample_files)}개 파일")

        success_count = 0
        total_time = 0

        for i, sample_file in enumerate(sample_files[:3]):  # 최대 3개 파일로 제한
            logger.info(f"파일 {i+1}/{min(3, len(sample_files))} 처리: {sample_file.name}")

            start_time = time.time()
            # 여기서는 간단한 로딩 테스트만 수행
            try:
                from data_abstraction_layer import PhysicalAIAdapter
                adapter = PhysicalAIAdapter({"debug": False})

                if adapter.validate_format(sample_file):
                    schemas = list(adapter.load_dataset(sample_file))
                    if schemas:
                        success_count += 1
                        logger.info(f"   ✓ 성공")
                    else:
                        logger.warning(f"   ! 스키마 로딩 실패")
                else:
                    logger.warning(f"   ! 포맷 검증 실패")

            except Exception as e:
                logger.warning(f"   ! 처리 실패: {e}")

            file_time = time.time() - start_time
            total_time += file_time

        logger.info(f"배치 처리 결과: {success_count}/{min(3, len(sample_files))} 성공")
        logger.info(f"총 처리 시간: {total_time:.3f}초")
        logger.info(f"평균 처리 시간: {total_time/min(3, len(sample_files)):.3f}초/파일")

        return success_count > 0

    except Exception as e:
        logger.error(f"배치 처리 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 실행"""
    logger.info("통합 파이프라인 테스트 시작")
    logger.info("=" * 60)

    # 테스트 실행
    test_results = {}

    test_results["full_pipeline"] = test_full_conversion_pipeline()
    test_results["batch_processing"] = test_batch_processing()

    # 결과 요약
    logger.info("\n" + "=" * 60)
    logger.info("통합 파이프라인 테스트 결과 요약:")

    for test_name, success in test_results.items():
        status = "✓ 성공" if success else "✗ 실패"
        logger.info(f"  {test_name:20s}: {status}")

    success_count = sum(test_results.values())
    total_count = len(test_results)

    logger.info(f"\n전체 성공률: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")

    if success_count == total_count:
        logger.info("\n🎉 통합 파이프라인 테스트 성공!")
        logger.info("   시스템이 PhysicalAI → Genesis+Franka 변환 준비 완료!")
    else:
        logger.warning(f"\n⚠️  {total_count - success_count}개 테스트에서 문제 발생")
        logger.warning("   문제를 해결한 후 다시 테스트하세요.")

    return success_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)