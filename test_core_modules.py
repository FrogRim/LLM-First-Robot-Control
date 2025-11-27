#!/usr/bin/env python3
"""
핵심 모듈들의 기본 동작 테스트

설계한 각 계층의 모듈들이 정상적으로 동작하는지 확인합니다.
"""

import sys
import traceback
import logging
from pathlib import Path
import json

# 프로젝트 모듈 임포트
sys.path.append('src')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_abstraction_layer():
    """데이터 추상화 계층 테스트"""
    logger.info("=== 데이터 추상화 계층 테스트 ===")

    try:
        from data_abstraction_layer import (
            PhysicalAIAdapter, StandardDataSchema, PhysicalProperties,
            RobotConfiguration, TrajectoryData, SceneDescription,
            ProcessingMetadata, AdapterFactory, DataValidator
        )

        logger.info("1. 모듈 임포트 성공")

        # PhysicalAI 어댑터 생성
        config = {"debug": True}
        adapter = PhysicalAIAdapter(config)
        logger.info("2. PhysicalAI 어댑터 생성 성공")

        # 샘플 데이터 파일 경로 (가장 최신 파일 선택)
        sample_files = list(Path("data/test_samples").glob("*.json"))
        if not sample_files:
            logger.warning("샘플 JSON 파일이 없습니다. 먼저 generate_sample_data.py를 실행하세요.")
            return False

        # 가장 최신 파일 선택 (수정 시간 기준)
        sample_file = max(sample_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"3. 샘플 파일 사용: {sample_file}")

        # 데이터 포맷 검증
        is_valid = adapter.validate_format(sample_file)
        logger.info(f"4. 데이터 포맷 검증: {'성공' if is_valid else '실패'}")

        if not is_valid:
            logger.error("데이터 포맷이 유효하지 않습니다.")
            return False

        # 메타데이터 추출
        metadata = adapter.get_metadata(sample_file)
        logger.info(f"5. 메타데이터 추출 성공: {len(metadata)} 항목")

        # 데이터 로딩 테스트
        logger.info("6. 데이터 로딩 테스트...")
        try:
            schemas = list(adapter.load_dataset(sample_file))
            logger.info(f"   로딩된 스키마 수: {len(schemas)}")

            if schemas:
                schema = schemas[0]
                logger.info(f"   스키마 타입: {type(schema)}")
                logger.info(f"   로봇 타입: {schema.robot_config.robot_type if hasattr(schema, 'robot_config') else 'N/A'}")
        except Exception as e:
            logger.error(f"   데이터 로딩 실패: {e}")
            return False

        # 데이터 검증기 테스트
        logger.info("7. 데이터 검증기 테스트...")
        validator = DataValidator(config)

        if schemas:
            validation_result = validator.validate_schema(schemas[0])
            logger.info(f"   검증 결과: {'성공' if validation_result['is_valid'] else '실패'}")
            logger.info(f"   완전성 점수: {validation_result['completeness_score']:.2f}")

        logger.info("✓ 데이터 추상화 계층 테스트 완료")
        return True

    except Exception as e:
        logger.error(f"✗ 데이터 추상화 계층 테스트 실패: {e}")
        traceback.print_exc()
        return False

def test_physics_mapping_layer():
    """물리 매핑 계층 테스트"""
    logger.info("\n=== 물리 매핑 계층 테스트 ===")

    try:
        from physics_mapping_layer import (
            UniversalPhysicsMapper, TrajectoryProcessor, UnitSystemNormalizer,
            PhysicsConstants
        )
        from data_abstraction_layer import (
            PhysicalProperties, PhysicsEngine, RobotConfiguration, RobotType,
            TrajectoryData
        )
        import numpy as np

        logger.info("1. 모듈 임포트 성공")

        # 물리 매퍼 생성
        config = {"debug": True}
        physics_mapper = UniversalPhysicsMapper(config)
        logger.info("2. 물리 매퍼 생성 성공")

        # 테스트용 물리 속성 생성
        source_props = PhysicalProperties(
            mass=2.5,
            friction_coefficient=0.7,
            restitution=0.3,
            linear_damping=0.1,
            angular_damping=0.1,
            material_type="aluminum"
        )
        logger.info("3. 테스트용 물리 속성 생성")

        # 물리 속성 변환 테스트
        logger.info("4. 물리 속성 변환 테스트...")
        converted_props = physics_mapper.map_properties(
            source_props,
            PhysicsEngine.ISAAC_SIM,
            PhysicsEngine.GENESIS_AI
        )

        logger.info(f"   원본 마찰계수: {source_props.friction_coefficient}")
        logger.info(f"   변환된 마찰계수: {converted_props.friction_coefficient}")
        logger.info(f"   질량 보존: {abs(converted_props.mass - source_props.mass) < 0.01}")

        # 변환 검증
        logger.info("5. 변환 검증 테스트...")
        validation_result = physics_mapper.validate_conversion(source_props, converted_props)
        logger.info(f"   검증 통과: {validation_result['is_valid']}")
        logger.info(f"   정확도 점수: {validation_result['accuracy_score']:.3f}")

        # 궤적 처리기 테스트
        logger.info("6. 궤적 처리기 테스트...")
        trajectory_processor = TrajectoryProcessor(config)

        # 테스트용 궤적 데이터
        timestamps = np.linspace(0, 5, 500)  # 5초, 100Hz
        joint_positions = np.random.rand(500, 7)  # 7-DOF

        test_trajectory = TrajectoryData(
            timestamps=timestamps,
            joint_positions=joint_positions,
            sampling_rate=100.0
        )

        # 리샘플링 테스트
        resampled = trajectory_processor.resample_trajectory(test_trajectory, target_frequency=50.0)
        logger.info(f"   원본 길이: {len(test_trajectory.timestamps)}")
        logger.info(f"   리샘플링 후: {len(resampled.timestamps)}")

        # 단위 정규화 테스트
        logger.info("7. 단위 정규화 테스트...")
        normalizer = UnitSystemNormalizer()

        source_units = {"mass": "kg", "length": "m", "angle": "rad"}
        normalized_props = normalizer.normalize_physical_properties(source_props, source_units)

        logger.info(f"   정규화 완료: {normalized_props.mass} kg")

        logger.info("✓ 물리 매핑 계층 테스트 완료")
        return True

    except Exception as e:
        logger.error(f"✗ 물리 매핑 계층 테스트 실패: {e}")
        traceback.print_exc()
        return False

def test_language_generation_layer():
    """언어 생성 계층 테스트"""
    logger.info("\n=== 언어 생성 계층 테스트 ===")

    try:
        from language_generation_layer import (
            TemplateEngine, QualityAssuranceSystem, GeneratedAnnotation,
            LanguageComplexity, AnnotationType
        )
        from data_abstraction_layer import (
            StandardDataSchema, PhysicalProperties, RobotConfiguration,
            TrajectoryData, SceneDescription, ProcessingMetadata,
            PhysicsEngine, RobotType
        )
        import numpy as np

        logger.info("1. 모듈 임포트 성공")

        # 템플릿 엔진 생성
        config = {"debug": True}
        template_engine = TemplateEngine(config)
        logger.info("2. 템플릿 엔진 생성 성공")

        # 테스트용 표준 스키마 생성
        robot_config = RobotConfiguration(
            robot_type=RobotType.FRANKA_PANDA,
            joint_count=7,
            joint_names=[f"joint_{i}" for i in range(7)],
            joint_types=["revolute"] * 7,
            joint_limits={f"joint_{i}": (-2.8, 2.8) for i in range(7)},
            base_pose=np.eye(4),
            end_effector_offset=np.array([0, 0, 0.1]),
            link_lengths=[0.333, 0.316, 0.384, 0.0825, 0.384, 0.088, 0.107],
            link_masses=[4.970, 0.646, 3.228, 3.587, 1.225, 1.666, 0.735],
            joint_stiffness=[3000.0] * 7,
            joint_damping=[100.0] * 7,
            max_joint_velocities=[2.175] * 7,
            max_joint_torques=[87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]
        )

        physical_props = PhysicalProperties(
            mass=1.5,
            friction_coefficient=0.8,
            restitution=0.2,
            linear_damping=0.1,
            angular_damping=0.1,
            material_type="plastic"
        )

        timestamps = np.linspace(0, 3, 300)
        trajectory = TrajectoryData(
            timestamps=timestamps,
            joint_positions=np.random.rand(300, 7),
            sampling_rate=100.0
        )

        scene = SceneDescription(
            task_type="manipulation",
            task_description="pick and place task",
            environment_type="laboratory"
        )

        metadata = ProcessingMetadata(
            source_engine=PhysicsEngine.ISAAC_SIM,
            target_engine=PhysicsEngine.GENESIS_AI,
            conversion_timestamp="2025-01-01T00:00:00",
            conversion_version="1.0.0",
            data_completeness=1.0,
            physics_accuracy=0.95,
            language_quality=0.85,
            processing_time=1.5,
            memory_usage=256.0
        )

        test_schema = StandardDataSchema(
            robot_config=robot_config,
            physical_properties=physical_props,
            trajectory_data=trajectory,
            scene_description=scene,
            metadata=metadata
        )

        logger.info("3. 테스트용 표준 스키마 생성 완료")

        # 실행 명령 생성 테스트
        logger.info("4. 실행 명령 생성 테스트...")
        instruction = template_engine.generate_instruction(
            test_schema,
            complexity=LanguageComplexity.INTERMEDIATE
        )
        logger.info(f"   생성된 명령: {instruction.text}")
        logger.info(f"   신뢰도: {instruction.confidence:.3f}")

        # 행동 설명 생성 테스트
        logger.info("5. 행동 설명 생성 테스트...")
        description = template_engine.generate_description(
            test_schema,
            complexity=LanguageComplexity.INTERMEDIATE
        )
        logger.info(f"   생성된 설명: {description.text}")

        # 물리 설명 생성 테스트
        logger.info("6. 물리 설명 생성 테스트...")
        explanation = template_engine.generate_explanation(
            test_schema,
            physics_focus=["mass", "friction"]
        )
        logger.info(f"   생성된 설명: {explanation.text}")

        # 품질 보증 시스템 테스트
        logger.info("7. 품질 보증 시스템 테스트...")
        qa_system = QualityAssuranceSystem(config)

        # 각 어노테이션의 품질 평가
        annotations = [instruction, description, explanation]
        for i, annotation in enumerate(annotations):
            metrics = qa_system.assess_quality(annotation)
            validation = qa_system.validate_annotation(annotation)

            logger.info(f"   어노테이션 {i+1}:")
            logger.info(f"     전체 품질: {metrics.overall_score:.3f}")
            logger.info(f"     검증 통과: {validation['is_valid']}")

        logger.info("✓ 언어 생성 계층 테스트 완료")
        return True

    except Exception as e:
        logger.error(f"✗ 언어 생성 계층 테스트 실패: {e}")
        traceback.print_exc()
        return False

def test_config_monitoring_system():
    """설정 관리 및 모니터링 시스템 테스트"""
    logger.info("\n=== 설정 관리 및 모니터링 시스템 테스트 ===")

    try:
        from config_monitoring_system import (
            ConfigurationManager, MetricsCollector, MonitoringDashboard,
            ConfigScope, ConfigPriority
        )

        logger.info("1. 모듈 임포트 성공")

        # 설정 관리자 테스트
        logger.info("2. 설정 관리자 테스트...")
        config_manager = ConfigurationManager()

        # 기본 설정 조회
        batch_size = config_manager.get_config("data.batch_size", 32)
        quality_threshold = config_manager.get_config("language.quality_threshold", 0.7)

        logger.info(f"   배치 크기: {batch_size}")
        logger.info(f"   품질 임계값: {quality_threshold}")

        # 런타임 설정 변경
        config_manager.set_config("test.parameter", 42, ConfigScope.RUNTIME, ConfigPriority.RUNTIME, "test")
        test_value = config_manager.get_config("test.parameter")
        logger.info(f"   테스트 설정: {test_value}")

        # 메트릭 수집기 테스트
        logger.info("3. 메트릭 수집기 테스트...")
        metrics_collector = MetricsCollector(config_manager)

        # 테스트 메트릭 기록
        metrics_collector.record_metric("test.processing_time", 1.23, unit="seconds")
        metrics_collector.increment_counter("test.processed_items", 5)

        # 최신 메트릭 조회
        latest_metric = metrics_collector.get_latest_metric("test.processing_time")
        if latest_metric:
            logger.info(f"   처리 시간: {latest_metric.value} {latest_metric.unit}")

        # 모니터링 대시보드 테스트
        logger.info("4. 모니터링 대시보드 테스트...")
        dashboard = MonitoringDashboard(config_manager, metrics_collector)

        # 시스템 상태 조회
        system_status = dashboard.get_system_status()
        logger.info(f"   시스템 상태 키: {list(system_status.keys())}")

        logger.info("✓ 설정 관리 및 모니터링 시스템 테스트 완료")
        return True

    except Exception as e:
        logger.error(f"✗ 설정 관리 및 모니터링 시스템 테스트 실패: {e}")
        traceback.print_exc()
        return False

def main():
    """메인 테스트 실행"""
    logger.info("핵심 모듈 기본 동작 테스트 시작")
    logger.info("=" * 60)

    # 각 계층별 테스트 실행
    test_results = {}

    test_results["data_abstraction"] = test_data_abstraction_layer()
    test_results["physics_mapping"] = test_physics_mapping_layer()
    test_results["language_generation"] = test_language_generation_layer()
    test_results["config_monitoring"] = test_config_monitoring_system()

    # 결과 요약
    logger.info("\n" + "=" * 60)
    logger.info("핵심 모듈 테스트 결과 요약:")

    for module_name, success in test_results.items():
        status = "✓ 성공" if success else "✗ 실패"
        logger.info(f"  {module_name:20s}: {status}")

    success_count = sum(test_results.values())
    total_count = len(test_results)

    logger.info(f"\n전체 성공률: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")

    if success_count == total_count:
        logger.info("\n🎉 모든 핵심 모듈 테스트 성공!")
        logger.info("   통합 파이프라인 테스트를 진행할 수 있습니다.")
    else:
        logger.warning(f"\n⚠️  {total_count - success_count}개 모듈에서 문제 발생")
        logger.warning("   문제를 해결한 후 통합 테스트를 진행하세요.")

    return success_count == total_count

if __name__ == "__main__":
    success = main()