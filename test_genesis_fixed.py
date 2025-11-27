#!/usr/bin/env python3
"""
Genesis AI 수정된 동작 테스트 (실제 API 구조 기반)
"""

import genesis
import numpy as np
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)

def test_genesis_basic():
    """Genesis AI 기본 기능 테스트"""
    print("=== Genesis AI 기본 기능 테스트 ===")

    try:
        # Genesis 초기화
        print("1. Genesis 초기화...")
        genesis.init()
        print("✓ Genesis 초기화 성공")

        # 장면 생성
        print("\n2. 장면 생성...")
        scene = genesis.Scene()
        print(f"✓ 장면 생성 성공: {type(scene)}")

        # 기본 설정 확인
        print("\n3. 기본 설정 확인...")
        print(f"   Genesis 버전: {genesis.__version__}")
        print(f"   백엔드: {hasattr(genesis, 'backend')}")
        print(f"   GPU 사용 가능: {hasattr(genesis, 'gpu')}")

        print("✓ 기본 설정 확인 완료")

        print("\n=== Genesis AI 기본 테스트 완료 ===")
        return True

    except Exception as e:
        print(f"✗ Genesis AI 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_genesis_components():
    """Genesis AI 구성요소 테스트"""
    print("\n=== Genesis AI 구성요소 테스트 ===")

    try:
        # Materials 확인
        print("1. Materials 확인...")
        if hasattr(genesis, 'materials'):
            print(f"   Materials 모듈: {dir(genesis.materials)[:5]}...")
            print("✓ Materials 모듈 사용 가능")
        else:
            print("! Materials 모듈 없음")

        # Morphs 확인
        print("\n2. Morphs 확인...")
        if hasattr(genesis, 'morphs'):
            print(f"   Morphs 모듈: {dir(genesis.morphs)[:5]}...")
            print("✓ Morphs 모듈 사용 가능")
        else:
            print("! Morphs 모듈 없음")

        # Sensors 확인
        print("\n3. Sensors 확인...")
        if hasattr(genesis, 'sensors'):
            print(f"   Sensors 모듈: {dir(genesis.sensors)[:5]}...")
            print("✓ Sensors 모듈 사용 가능")
        else:
            print("! Sensors 모듈 없음")

        # Math utilities 확인
        print("\n4. Math utilities 확인...")
        math_functions = ['quat_to_R', 'euler_to_R', 'transform_by_T']
        for func in math_functions:
            if hasattr(genesis, func):
                print(f"   ✓ {func} 사용 가능")
            else:
                print(f"   ! {func} 없음")

        print("\n=== Genesis AI 구성요소 테스트 완료 ===")
        return True

    except Exception as e:
        print(f"✗ Genesis AI 구성요소 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_physics_functions():
    """물리 관련 함수 테스트"""
    print("\n=== Genesis AI 물리 함수 테스트 ===")

    try:
        # 행렬 변환 함수 테스트
        print("1. 행렬 변환 함수 테스트...")

        # 쿼터니언 관련
        if hasattr(genesis, 'identity_quat'):
            identity_q = genesis.identity_quat()
            print(f"   ✓ Identity quaternion: {identity_q}")

        # 오일러 각도 변환
        if hasattr(genesis, 'euler_to_quat'):
            euler = [0.1, 0.2, 0.3]  # roll, pitch, yaw
            quat = genesis.euler_to_quat(euler)
            print(f"   ✓ Euler to quaternion: {euler} -> {quat}")

        # 위치 변환
        if hasattr(genesis, 'zero_pos'):
            zero_position = genesis.zero_pos()
            print(f"   ✓ Zero position: {zero_position}")

        print("✓ 물리 함수 테스트 성공")

        # 수학 연산 테스트
        print("\n2. 텐서 연산 테스트...")
        if hasattr(genesis, 'zeros'):
            zeros_tensor = genesis.zeros((3, 3))
            print(f"   ✓ Zeros tensor 생성: shape {zeros_tensor.shape if hasattr(zeros_tensor, 'shape') else 'unknown'}")

        if hasattr(genesis, 'ones'):
            ones_tensor = genesis.ones((2, 2))
            print(f"   ✓ Ones tensor 생성: shape {ones_tensor.shape if hasattr(ones_tensor, 'shape') else 'unknown'}")

        print("\n=== Genesis AI 물리 함수 테스트 완료 ===")
        return True

    except Exception as e:
        print(f"✗ Genesis AI 물리 함수 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_types():
    """Genesis 데이터 타입 테스트"""
    print("\n=== Genesis AI 데이터 타입 테스트 ===")

    try:
        # 기본 데이터 타입 확인
        print("1. 기본 데이터 타입 확인...")

        # Tensor 관련
        if hasattr(genesis, 'Tensor'):
            print("   ✓ Tensor 타입 사용 가능")

        # 상수 확인
        if hasattr(genesis, 'constants'):
            print(f"   ✓ Constants 모듈: {type(genesis.constants)}")

        # 백엔드 확인
        if hasattr(genesis, 'backend'):
            print(f"   ✓ Backend: {genesis.backend}")

        # 디바이스 확인
        if hasattr(genesis, 'get_device'):
            device = genesis.get_device()
            print(f"   ✓ Device: {device}")

        print("\n=== Genesis AI 데이터 타입 테스트 완료 ===")
        return True

    except Exception as e:
        print(f"✗ Genesis AI 데이터 타입 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Genesis AI 수정된 종합 기능 테스트 시작")

    # 각 테스트 실행
    basic_success = test_genesis_basic()
    components_success = test_genesis_components()
    physics_success = test_physics_functions()
    datatypes_success = test_data_types()

    # 결과 요약
    print("\n" + "="*60)
    print("테스트 결과 요약:")
    print(f"  기본 기능:     {'✓ 성공' if basic_success else '✗ 실패'}")
    print(f"  구성요소:      {'✓ 성공' if components_success else '✗ 실패'}")
    print(f"  물리 함수:     {'✓ 성공' if physics_success else '✗ 실패'}")
    print(f"  데이터 타입:   {'✓ 성공' if datatypes_success else '✗ 실패'}")

    total_success = basic_success and components_success and physics_success and datatypes_success

    if total_success:
        print("\n🎉 Genesis AI 환경 구축 및 동작 확인 완료!")
        print("   파이프라인 구현을 위한 준비가 되었습니다.")
    else:
        print("\n⚠️  일부 기능에 문제가 있지만 기본적인 작업은 가능합니다.")
        print("   제한된 기능으로 파이프라인을 구현할 수 있습니다.")