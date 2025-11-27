#!/usr/bin/env python3
"""
LoRA 어댑터 병합 스크립트 - CPU 전용 버전

특징:
1. CPU 전용 동작 (device_map={"": "cpu"})
2. 32GB RAM으로 안정적 병합
3. 진행 상황 상세 표시
4. FP16 저장 (메모리 절약)

예상 소요 시간: 15-30분
메모리 요구사항: 32GB RAM (현재 시스템으로 충분)
"""

import os
import sys
import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def print_memory_usage():
    """현재 메모리 사용량 출력"""
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        mem_gb = mem_info.rss / (1024 ** 3)
        print(f"    💾 현재 RAM 사용량: {mem_gb:.2f} GB")
    except:
        pass


def merge_lora_adapter_cpu(
    base_model_name: str,
    adapter_dir: str,
    output_dir: str,
    save_fp16: bool = True
):
    """
    LoRA 어댑터를 베이스 모델에 병합 (CPU 전용)

    Args:
        base_model_name: 베이스 모델 이름
        adapter_dir: LoRA 어댑터 디렉토리
        output_dir: 출력 디렉토리
        save_fp16: FP16으로 저장 (메모리 절약)
    """

    print("="*70)
    print("🔧 LoRA 어댑터 병합 시작 (CPU 전용 모드)")
    print("="*70)
    print(f"\n📋 설정:")
    print(f"  - 베이스 모델: {base_model_name}")
    print(f"  - 어댑터 경로: {adapter_dir}")
    print(f"  - 출력 경로: {output_dir}")
    print(f"  - 저장 형식: {'FP16' if save_fp16 else 'FP32'}")
    print(f"  - 디바이스: CPU (안정적, 느림)")
    print(f"\n⏱️  예상 소요 시간: 15-30분")
    print(f"    (CPU 전용이므로 느리지만, 안정적으로 완료됩니다)")

    print_memory_usage()

    # 어댑터 존재 확인
    if not os.path.exists(adapter_dir):
        print(f"\n❌ 오류: 어댑터 디렉토리가 존재하지 않습니다: {adapter_dir}")
        return False

    print(f"\n{'='*70}")
    print("시작 시간:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print('='*70)

    # Step 1: 베이스 모델 로드
    print("\n📥 Step 1/5: 베이스 모델 로드 중...")
    print("    💡 CPU 전용 모드로 로드합니다")
    print("    ⏱️  예상 시간: 5-10분")

    step_start = time.time()

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map={"": "cpu"},  # 🔑 CPU 전용
            trust_remote_code=True,
            torch_dtype=torch.float16 if save_fp16 else torch.float32,
            low_cpu_mem_usage=True,  # CPU 메모리 최적화
        )

        step_time = time.time() - step_start
        print(f"    ✅ 베이스 모델 로드 완료 ({step_time/60:.1f}분 소요)")
        print_memory_usage()

    except Exception as e:
        print(f"    ❌ 베이스 모델 로드 실패: {e}")
        return False

    # Step 2: LoRA 어댑터 로드
    print("\n📥 Step 2/5: LoRA 어댑터 로드 중...")
    print("    ⏱️  예상 시간: 1-2분")

    step_start = time.time()

    try:
        model = PeftModel.from_pretrained(base_model, adapter_dir)

        step_time = time.time() - step_start
        print(f"    ✅ LoRA 어댑터 로드 완료 ({step_time:.1f}초 소요)")
        print_memory_usage()

    except Exception as e:
        print(f"    ❌ LoRA 어댑터 로드 실패: {e}")
        return False

    # Step 3: 병합
    print("\n🔨 Step 3/5: LoRA 어댑터를 베이스 모델에 병합 중...")
    print("    💡 어댑터 가중치를 베이스 모델에 통합합니다")
    print("    ⏱️  예상 시간: 2-5분")

    step_start = time.time()

    try:
        merged_model = model.merge_and_unload()

        step_time = time.time() - step_start
        print(f"    ✅ 병합 완료! ({step_time/60:.1f}분 소요)")
        print_memory_usage()

    except Exception as e:
        print(f"    ❌ 병합 실패: {e}")
        return False

    # Step 4: 병합된 모델 저장
    print(f"\n💾 Step 4/5: 병합된 모델 저장 중...")
    print(f"    📁 경로: {output_dir}")
    print(f"    💡 safetensors 형식으로 저장합니다")
    print("    ⏱️  예상 시간: 5-10분")

    step_start = time.time()

    try:
        os.makedirs(output_dir, exist_ok=True)

        # 진행 상황 표시를 위해 작은 콜백 추가
        print("    📝 모델 가중치 저장 중...")
        merged_model.save_pretrained(
            output_dir,
            safe_serialization=True,  # safetensors 사용
        )

        step_time = time.time() - step_start
        print(f"    ✅ 모델 저장 완료 ({step_time/60:.1f}분 소요)")

    except Exception as e:
        print(f"    ❌ 모델 저장 실패: {e}")
        return False

    # Step 5: 토크나이저 저장
    print(f"\n💾 Step 5/5: 토크나이저 저장 중...")
    print("    ⏱️  예상 시간: 10초 미만")

    step_start = time.time()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(output_dir)

        step_time = time.time() - step_start
        print(f"    ✅ 토크나이저 저장 완료 ({step_time:.1f}초 소요)")

    except Exception as e:
        print(f"    ❌ 토크나이저 저장 실패: {e}")
        return False

    # 성공!
    print("\n" + "="*70)
    print("🎉 LoRA 병합 완료!")
    print("="*70)
    print("완료 시간:", time.strftime("%Y-%m-%d %H:%M:%S"))

    # 저장된 파일 정보
    print(f"\n📁 저장된 파일:")
    try:
        total_size = 0
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                filepath = os.path.join(root, file)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                total_size += size_mb
                if size_mb > 1:  # 1MB 이상 파일만 표시
                    print(f"    - {file}: {size_mb:.1f} MB")

        total_gb = total_size / 1024
        print(f"\n📊 총 크기: {total_size:.1f} MB ({total_gb:.2f} GB)")

        if save_fp16:
            print(f"    💡 FP16 형식으로 저장되어 메모리 효율적입니다")

    except Exception as e:
        print(f"    (파일 크기 계산 실패: {e})")

    print(f"\n📌 다음 단계:")
    print(f"    1. 병합된 모델 테스트:")
    print(f"       python scripts/test_merged_model.py")
    print(f"    2. 추가 최적화 (선택):")
    print(f"       python scripts/quantize_int8.py")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="LoRA 어댑터 병합 (CPU 전용)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 기본 설정으로 병합
  python scripts/merge_lora_adapter_cpu.py

  # FP32로 저장 (더 높은 정밀도, 더 큰 파일)
  python scripts/merge_lora_adapter_cpu.py --fp32

  # 커스텀 경로 지정
  python scripts/merge_lora_adapter_cpu.py --adapter_dir ./my_adapter --output_dir ./my_output
        """
    )
    parser.add_argument(
        "--base_model",
        default="Qwen/Qwen2.5-14B-Instruct",
        help="베이스 모델 이름"
    )
    parser.add_argument(
        "--adapter_dir",
        default="./droid-physics-qwen14b-qlora",
        help="LoRA 어댑터 디렉토리"
    )
    parser.add_argument(
        "--output_dir",
        default="./droid-physics-qwen14b-merged",
        help="출력 디렉토리"
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="FP32로 저장 (기본: FP16)"
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="확인 프롬프트 없이 바로 실행"
    )
    args = parser.parse_args()

    # 시작 확인
    if not args.yes:
        print("\n⚠️  주의사항:")
        print("  - 이 작업은 15-30분 소요됩니다")
        print("  - CPU 전용 모드로 안전하게 동작합니다")
        print("  - 작업 중 프로그램을 종료하지 마세요")

        response = input("\n계속하시겠습니까? (y/N): ")
        if response.lower() != 'y':
            print("작업이 취소되었습니다.")
            return

    start_time = time.time()

    success = merge_lora_adapter_cpu(
        base_model_name=args.base_model,
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        save_fp16=not args.fp32
    )

    elapsed_time = time.time() - start_time

    if success:
        print(f"\n✅ 모든 작업 완료!")
        print(f"⏱️  총 소요 시간: {elapsed_time/60:.1f}분")
    else:
        print(f"\n❌ 작업 실패")
        print(f"⏱️  실패 시점: {elapsed_time/60:.1f}분 경과")
        sys.exit(1)


if __name__ == "__main__":
    main()
