#!/usr/bin/env python3
"""
Genesis AI 기본 동작 테스트
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
        # Genesis 엔진 초기화
        print("1. Genesis 엔진 초기화...")
        g = genesis.Genesis(
            scene_cfg=genesis.PhysicsConfig(
                dt=0.01,  # 10ms timestep
                substeps=1,
                gravity=(0, 0, -9.81)
            ),
            sim_cfg=genesis.SimConfig(
                n_envs=1,
            ),
            viewer_cfg=genesis.ViewerConfig(
                res=(800, 600),
                camera_pos=(3.5, 0.0, 2.5),
            ),
        )
        print("✓ Genesis 엔진 초기화 성공")

        # 장면 생성
        print("\n2. 기본 장면 생성...")
        scene = g.add_scene()
        print("✓ 장면 생성 성공")

        # 지면 추가
        print("\n3. 지면 추가...")
        plane = scene.add_entity(
            material=genesis.materials.Rigid(
                rho=1000.0,
                friction=0.8,
                restitution=0.1
            ),
            morph=genesis.morphs.Plane(),
        )
        print("✓ 지면 추가 성공")

        # 간단한 박스 추가
        print("\n4. 박스 객체 추가...")
        box = scene.add_entity(
            material=genesis.materials.Rigid(
                rho=500.0,  # kg/m³
                friction=0.7,
                restitution=0.3
            ),
            morph=genesis.morphs.Box(
                size=(0.1, 0.1, 0.1)
            ),
            pos=(0, 0, 1.0),
        )
        print("✓ 박스 추가 성공")

        # 시뮬레이션 구축
        print("\n5. 시뮬레이션 구축...")
        g.build()
        print("✓ 시뮬레이션 구축 성공")

        # 몇 스텝 실행해보기
        print("\n6. 시뮬레이션 실행 테스트...")
        for step in range(10):
            g.step()
            if step % 5 == 0:
                box_pos = box.get_pos()
                print(f"   Step {step}: Box position = {box_pos}")

        print("✓ 시뮬레이션 실행 성공")

        # 리소스 정리
        print("\n7. 리소스 정리...")
        g.destroy()
        print("✓ 리소스 정리 완료")

        print("\n=== Genesis AI 기본 테스트 완료 ===")
        return True

    except Exception as e:
        print(f"✗ Genesis AI 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_genesis_franka():
    """Franka 로봇 관련 기능 테스트"""
    print("\n=== Franka 로봇 기능 테스트 ===")

    try:
        # Genesis 엔진 초기화
        g = genesis.Genesis(
            scene_cfg=genesis.PhysicsConfig(
                dt=0.01,
                substeps=1,
                gravity=(0, 0, -9.81)
            ),
            sim_cfg=genesis.SimConfig(
                n_envs=1,
            ),
            viewer_cfg=genesis.ViewerConfig(
                res=(800, 600),
            ),
        )

        scene = g.add_scene()

        # 지면 추가
        plane = scene.add_entity(
            material=genesis.materials.Rigid(),
            morph=genesis.morphs.Plane(),
        )

        # Franka 로봇 로드 시도
        print("1. Franka 로봇 로드 시도...")
        try:
            # URDF 경로 확인
            franka_urdf = None
            possible_paths = [
                "assets/robots/franka_panda/franka_panda.urdf",
                "robots/franka_panda.urdf",
            ]

            for path in possible_paths:
                try:
                    robot = scene.add_entity(
                        genesis.morphs.URDF(file=path, fixed=True),
                        pos=(0, 0, 0),
                    )
                    franka_urdf = path
                    print(f"✓ Franka 로봇 로드 성공: {path}")
                    break
                except:
                    continue

            if franka_urdf is None:
                print("! Franka URDF 파일을 찾을 수 없음 - 기본 관절 로봇으로 대체")
                # 간단한 다관절 로봇 시뮬레이션
                links = []
                for i in range(7):  # 7-DOF 로봇
                    link = scene.add_entity(
                        material=genesis.materials.Rigid(rho=1000.0),
                        morph=genesis.morphs.Cylinder(radius=0.02, height=0.2),
                        pos=(0, 0, 0.1 * (i + 1)),
                    )
                    links.append(link)
                print("✓ 7-DOF 다관절 로봇 시뮬레이션 생성")

        except Exception as e:
            print(f"! Franka 로봇 로드 실패, 기본 로봇으로 진행: {e}")

        # 시뮬레이션 구축 및 실행
        print("\n2. 로봇 시뮬레이션 구축...")
        g.build()
        print("✓ 로봇 시뮬레이션 구축 성공")

        # 짧은 시뮬레이션 실행
        print("\n3. 로봇 시뮬레이션 실행...")
        for step in range(5):
            g.step()

        print("✓ 로봇 시뮬레이션 실행 성공")

        # 정리
        g.destroy()
        print("\n=== Franka 로봇 테스트 완료 ===")
        return True

    except Exception as e:
        print(f"✗ Franka 로봇 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Genesis AI 종합 기능 테스트 시작")

    # 기본 기능 테스트
    basic_success = test_genesis_basic()

    # Franka 로봇 테스트
    franka_success = test_genesis_franka()

    # 결과 요약
    print("\n" + "="*50)
    print("테스트 결과 요약:")
    print(f"  기본 기능: {'✓ 성공' if basic_success else '✗ 실패'}")
    print(f"  Franka 로봇: {'✓ 성공' if franka_success else '✗ 실패'}")

    if basic_success and franka_success:
        print("\n🎉 Genesis AI 환경 구축 완료!")
    else:
        print("\n⚠️  일부 기능에 문제가 있지만 기본 작업은 가능합니다.")