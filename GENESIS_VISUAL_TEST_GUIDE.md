# Genesis AI 시각적 테스트 가이드

**학습된 Qwen2.5-14B 모델을 Genesis AI와 통합하여 시각적으로 테스트하는 전체 가이드**

---

## 목차

1. [개요](#개요)
2. [아키텍처](#아키텍처)
3. [환경 설정](#환경-설정)
4. [빠른 시작](#빠른-시작)
5. [사용 방법](#사용-방법)
6. [테스트 시나리오](#테스트-시나리오)
7. [결과 분석](#결과-분석)
8. [트러블슈팅](#트러블슈팅)

---

## 개요

### 무엇을 하는가?

자연어 명령을 입력하면:
1. **LLM (Qwen2.5-14B QLora)**: 물리 파라미터 JSON 생성
2. **Genesis AI**: 해당 파라미터로 3D 물리 시뮬레이션 실행
3. **시각화**: 실시간으로 로봇 동작 확인

### 핵심 기능

- ✅ **자연어 → 물리 파라미터**: "Pick up the plastic bottle gently" → JSON
- ✅ **실시간 시각화**: Genesis AI 뷰어로 3D 시뮬레이션
- ✅ **다양한 재질**: 플라스틱, 금속, 유리, 나무, 고무
- ✅ **한국어 지원**: "유리병을 천천히 들어 올리세요"
- ✅ **성능 분석**: 추론 시간, 성공률 자동 측정

---

## 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                       사용자 입력                                │
│         "Pick up the plastic bottle gently"                     │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│              LLM (Qwen2.5-14B QLora)                            │
│  - 모델: droid-physics-qwen14b-qlora                            │
│  - 추론 시간: ~75ms                                              │
│  - JSON 파싱률: 100%                                             │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                   물리 파라미터 JSON                             │
│  {                                                               │
│    "physical_analysis": {                                        │
│      "material_inference": "plastic",                            │
│      "mass_category": "light",                                   │
│      "friction_coefficient": "medium",                           │
│      "grip_force": 0.3                                           │
│    }                                                             │
│  }                                                               │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                Genesis AI 시뮬레이션                             │
│  - 물리 엔진: 실시간 강체 동역학                                 │
│  - 렌더링: 1280×720 @ 60 FPS                                    │
│  - 객체 생성: 재질별 물리 속성 적용                              │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                   시각적 결과                                    │
│  - 3D 뷰어 창: 실시간 시뮬레이션 표시                            │
│  - 콘솔 출력: 물리 파라미터, 추론 시간                           │
│  - JSON 파일: 전체 결과 저장                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 환경 설정

### 1. 필수 요구사항

- **GPU**: NVIDIA GPU (VRAM ≥8GB 권장, RTX 3060 이상)
- **CUDA**: 11.8 이상
- **Python**: 3.10+
- **학습된 모델**: `droid-physics-qwen14b-qlora/` 디렉토리

### 2. Genesis AI 설치

```bash
# Genesis AI 설치 (공식 문서 참고)
pip install genesis-world
```

**참고**: Genesis AI는 GPU가 필요합니다. 설치 문제가 있다면 `--no-genesis` 플래그로 LLM 추론만 테스트할 수 있습니다.

### 3. 의존성 확인

```bash
# 현재 환경 활성화
source .venv/bin/activate  # 또는 conda activate droid_qwen14b

# 필요 패키지 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import genesis as gs; print('Genesis AI: OK')" 2>/dev/null || echo "Genesis AI: Not installed"
```

---

## 빠른 시작

### 옵션 1: 기본 테스트 (LLM만, Genesis 없이)

Genesis AI 설치가 안 되어 있거나 GPU 없이 테스트하고 싶을 때:

```bash
# LLM 추론만 테스트
python visual_demo.py --no-genesis --mode basic
```

**출력 예시**:
```
🎬 시나리오: 가벼운 플라스틱 물체 - 부드럽게
📝 명령: Pick up the plastic bottle gently and place it in the container
[1/3] LLM 추론 중...
✓ LLM 추론 완료 (75.3ms)

생성된 물리 파라미터:
{
  "material_inference": "plastic",
  "mass_category": "light",
  "friction_coefficient": "medium",
  "grip_force": 0.3
}
```

### 옵션 2: 전체 시뮬레이션 (Genesis AI 포함)

Genesis AI가 설치되어 있을 때:

```bash
# 기본 시나리오 (4개) - 시각화 포함
python visual_demo.py --mode basic

# 고급 시나리오 (4개) - 한국어 포함
python visual_demo.py --mode advanced

# 스트레스 테스트 (3개)
python visual_demo.py --mode stress

# 전체 실행 (11개 시나리오)
python visual_demo.py --mode all
```

**Genesis AI 뷰어가 열리면서 3D 시뮬레이션이 실행됩니다!**

### 옵션 3: 단일 명령 테스트

```bash
# 통합 스크립트를 직접 실행
python llm_genesis_integration.py
```

---

## 사용 방법

### 1. 기본 사용 - 데모 스크립트

```bash
python visual_demo.py [옵션]
```

**주요 옵션**:

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--mode` | 실행 모드 (basic/advanced/stress/all) | `basic` |
| `--no-genesis` | Genesis AI 비활성화 (LLM만) | False |
| `--adapter-dir` | QLora 어댑터 경로 | `./droid-physics-qwen14b-qlora` |
| `--output-dir` | 결과 저장 디렉토리 | `./demo_results` |

**예시**:

```bash
# 기본 4개 시나리오만 실행
python visual_demo.py --mode basic

# 고급 시나리오 + Genesis AI 비활성화
python visual_demo.py --mode advanced --no-genesis

# 전체 시나리오 + 다른 모델 사용
python visual_demo.py --mode all --adapter-dir ./my-custom-adapter
```

### 2. Python 코드에서 사용

```python
from llm_genesis_integration import LLMGenesisIntegration

# 초기화
integration = LLMGenesisIntegration(
    adapter_dir="./droid-physics-qwen14b-qlora",
    enable_genesis=True
)

# 명령 실행
result = integration.execute_command(
    command="Pick up the plastic bottle gently",
    simulate=True,
    duration_sec=2.0
)

# 결과 확인
print(result['params']['physical_analysis'])
print(f"추론 시간: {result['params']['_metadata']['inference_time_ms']:.1f}ms")

# 정리
integration.cleanup()
```

### 3. 커스텀 시나리오 추가

`visual_demo.py`를 수정하여 새로운 시나리오 추가:

```python
def run_my_custom_scenarios(self):
    """내 커스텀 시나리오"""
    scenarios = [
        {
            "name": "내 테스트 1",
            "command": "Pick up the red cube slowly",
            "duration": 2.0
        },
        # 더 추가...
    ]

    for scenario in scenarios:
        self.run_scenario(
            name=scenario["name"],
            command=scenario["command"],
            duration_sec=scenario["duration"]
        )
```

---

## 테스트 시나리오

### 기본 시나리오 (`--mode basic`)

| # | 이름 | 명령 | 특징 |
|---|------|------|------|
| 1 | 가벼운 플라스틱 | "Pick up the plastic bottle gently..." | 낮은 grip force |
| 2 | 무거운 금속 | "Grab the heavy metal tool firmly..." | 높은 grip force |
| 3 | 깨지기 쉬운 유리 | "Lift the glass cup very slowly..." | 낮은 속도 |
| 4 | 나무 블록 - 빠르게 | "Grab the wooden block quickly..." | 높은 속도 |

### 고급 시나리오 (`--mode advanced`)

| # | 이름 | 명령 | 특징 |
|---|------|------|------|
| 5 | 부드러운 고무공 | "Pick up the soft rubber ball..." | 높은 반발 계수 |
| 6 | 세라믹 머그 | "Carefully grasp the ceramic mug..." | 중간 물성 |
| 7 | 긴 금속 막대 | "Grab the long metal rod from the middle..." | 균형 제어 |
| 8 | 한국어 명령 | "유리병을 천천히 들어 올려서..." | 한국어 지원 |

### 스트레스 테스트 (`--mode stress`)

| # | 이름 | 명령 | 목적 |
|---|------|------|------|
| 9 | 최소 정보 | "Pick up the object" | 모호한 명령 처리 |
| 10 | 복잡한 명령 | "Pick up... rotate... place..." | 다단계 명령 |
| 11 | 애매한 명령 | "Move the thing over there..." | 불명확한 명령 |

---

## 결과 분석

### 1. 콘솔 출력

실행 중 실시간으로 표시:

```
📝 명령: Pick up the plastic bottle gently and place it in the container

[1/3] LLM 추론 중...
✓ LLM 추론 완료 (75.3ms)

생성된 물리 파라미터:
{
  "material_inference": "plastic",
  "mass_category": "light",
  "friction_coefficient": "medium",
  "fragility": "low",
  "grip_force": 0.3,
  "lift_speed": 0.4
}

[2/3] Genesis AI 객체 생성 중...
✓ Genesis AI 객체 생성: target_object
  - 재료: plastic, 밀도: 600.0 kg/m³
  - 마찰: 0.50, 반발: 0.20

[3/3] 시뮬레이션 실행 중...
🔨 시뮬레이션 구축 중...
✓ 시뮬레이션 구축 완료
▶️  시뮬레이션 실행 (2.0초)...
✓ 시뮬레이션 완료

✅ 명령 실행 완료
```

### 2. JSON 결과 파일

`demo_results/` 디렉토리에 저장:

```json
{
  "metadata": {
    "timestamp": "2025-10-13 18:30:00",
    "total_scenarios": 4
  },
  "analysis": {
    "total_scenarios": 4,
    "successful_inferences": 4,
    "successful_simulations": 4,
    "inference_success_rate": 1.0,
    "simulation_success_rate": 1.0,
    "avg_inference_time_ms": 75.3,
    "min_inference_time_ms": 68.2,
    "max_inference_time_ms": 82.7
  },
  "scenarios": [...]
}
```

### 3. 분석 메트릭

데모 완료 후 자동 출력:

```
📊 결과 분석

총 시나리오 수: 4
LLM 추론 성공: 4/4 (100.0%)
시뮬레이션 성공: 4/4 (100.0%)

평균 추론 시간: 75.3ms
최소 추론 시간: 68.2ms
최대 추론 시간: 82.7ms

💾 결과 저장: ./demo_results/demo_results_basic_20251013_183000.json
```

---

## 트러블슈팅

### 문제 1: Genesis AI 설치 오류

**증상**:
```
ImportError: No module named 'genesis'
```

**해결**:
```bash
# Genesis AI 없이 LLM 추론만 테스트
python visual_demo.py --no-genesis --mode basic
```

### 문제 2: CUDA Out of Memory

**증상**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**해결**:
- GPU 메모리가 부족한 경우 4-bit 양자화 사용 (이미 적용됨)
- 다른 GPU 프로세스 종료
- `batch_size` 줄이기 (코드에서 이미 1로 설정됨)

### 문제 3: 모델 파일을 찾을 수 없음

**증상**:
```
OSError: ./droid-physics-qwen14b-qlora not found
```

**해결**:
```bash
# 모델 경로 확인
ls -lh droid-physics-qwen14b-qlora/

# 다른 경로 지정
python visual_demo.py --adapter-dir /path/to/your/adapter
```

### 문제 4: Genesis AI 뷰어가 열리지 않음

**증상**:
- 시뮬레이션은 실행되지만 창이 안 뜸

**해결**:
- X11 forwarding 확인 (SSH 사용 시)
- 로컬 디스플레이 사용:
  ```bash
  export DISPLAY=:0
  python visual_demo.py
  ```
- 또는 headless 모드로 실행 (시각화 없이):
  ```bash
  python visual_demo.py --no-genesis
  ```

### 문제 5: JSON 파싱 실패

**증상**:
```
⚠️  JSON 파싱 실패: ...
```

**해결**:
- `scripts/json_sanitizer.py` 적용 (이미 학습에 반영됨)
- 낮은 temperature로 재시도 (기본값 0.1로 설정됨)
- 모델 재학습 필요 시 → `QWEN14B_TRAINING_GUIDE.md` 참고

---

## 다음 단계

### 1. 성능 벤치마크

```bash
# 추론 속도 측정
python scripts/benchmark_inference.py --adapter_dir ./droid-physics-qwen14b-qlora

# JSON 파싱 평가
python scripts/eval_physics_json.py --adapter_dir ./droid-physics-qwen14b-qlora
```

### 2. 실제 로봇 연동

- ROS2 인터페이스: `src/ros2_interface.py`
- 실시간 제어: `src/control_parameter_mapper.py`

### 3. 모델 개선

- 더 많은 학습 데이터: `scripts/build_v2_dataset.py`
- 하이퍼파라미터 튜닝: `droid_qlora_qwen14b.yml`
- 재학습: `QWEN14B_TRAINING_GUIDE.md`

---

## 관련 문서

- **학습 가이드**: `QWEN14B_TRAINING_GUIDE.md`
- **빠른 시작**: `QUICK_START_QWEN14B.md`
- **프로젝트 요약**: `FINAL_PROJECT_SUMMARY.md`
- **데이터 증강**: `DATA_AUGMENTATION_V2_SUMMARY.md`

---

## 문의 및 피드백

문제가 발생하거나 개선 제안이 있으시면:
- GitHub Issues
- 프로젝트 문서 참고
- 로그 파일 확인: `droid-physics-qwen14b-qlora/debug.log`

---

**🎉 Genesis AI 시각적 테스트를 즐기세요!**
