# 🤖 LLM Fine-tuning 가이드
## DROID 물리 도메인 학습 데이터로 LLM 학습하기

**프로젝트**: LLM-First 기반 물리 속성 추출 로봇 제어
**데이터**: DROID → Genesis AI + Franka 변환 데이터 (60 samples)
**목적**: LLM을 물리 도메인에 맞게 fine-tuning

---

## 📊 준비된 학습 데이터

### 생성 완료된 파일

| 파일명 | 포맷 | 용도 | 샘플 수 |
|--------|------|------|---------|
| `droid_physics_llm_train_alpaca.json` | Alpaca | 학습용 (권장) | 51 |
| `droid_physics_llm_test_alpaca.json` | Alpaca | 평가용 | 9 |
| `droid_physics_llm_train_sharegpt.json` | ShareGPT | 대화형 모델 | 51 |
| `droid_physics_llm_test_sharegpt.json` | ShareGPT | 대화형 평가 | 9 |
| `dataset_info.json` | Metadata | 데이터셋 정보 | - |

### 데이터 통계

- **총 샘플**: 60개 (Train: 51 / Test: 9)
- **재료 종류**: 7가지 (plastic, metal, glass, wood, rubber, ceramic, fabric)
- **명령어 변형**: 9가지 패턴
- **출력 형식**: JSON (physical_analysis + control_parameters + reasoning + affordance)

---

## 🚀 Phase 2-A: QLoRA Fine-tuning (권장)

### 왜 QLoRA인가?

- ✅ **메모리 효율적**: 4-bit quantization으로 24GB GPU 1장으로 7B 모델 학습 가능
- ✅ **빠른 학습**: Full fine-tuning 대비 4배 빠름
- ✅ **성능 유지**: LoRA 방식으로 성능 손실 최소화
- ✅ **졸업논문에 적합**: 개인 연구 환경에서 실행 가능

### 추천 베이스 모델

| 모델 | 크기 | 특징 | 메모리 요구 |
|------|------|------|-------------|
| **Mistral-7B-v0.1** | 7B | 최신, 성능 우수 (권장) | 24GB |
| **Llama-2-7b** | 7B | 안정적, 검증됨 | 24GB |
| **Phi-2** | 2.7B | 초경량, 빠른 테스트 | 8GB |

---

## 🛠️ 실제 학습 방법

### 옵션 1: Axolotl 사용 (가장 간단 ⭐)

#### 1. Axolotl 설치

```bash
# Conda 환경 생성
conda create -n droid_llm python=3.10
conda activate droid_llm

# Axolotl 설치
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
pip install packaging
pip install -e '.[flash-attn,deepspeed]'
```

#### 2. 학습 설정 파일 생성 (`droid_qlora.yml`)

```yaml
base_model: mistralai/Mistral-7B-v0.1
model_type: MistralForCausalLM
tokenizer_type: LlamaTokenizer

load_in_8bit: false
load_in_4bit: true
strict: false

datasets:
  - path: ./droid_physics_llm_train_alpaca.json
    type: alpaca

dataset_prepared_path: null
val_set_size: 0.1
output_dir: ./droid-physics-mistral-7b-qlora

adapter: qlora
lora_model_dir:

sequence_len: 2048
sample_packing: true
pad_to_sequence_len: true

lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_linear: true
lora_fan_in_fan_out:
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

wandb_project: droid-physics-llm
wandb_entity:
wandb_watch:
wandb_name: mistral-7b-qlora-physics
wandb_log_model:

gradient_accumulation_steps: 4
micro_batch_size: 2
num_epochs: 3
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
learning_rate: 0.0002

train_on_inputs: false
group_by_length: false
bf16: auto
fp16:
tf32: false

gradient_checkpointing: true
early_stopping_patience:
resume_from_checkpoint:
local_rank:
logging_steps: 10
xformers_attention:
flash_attention: true

warmup_steps: 10
evals_per_epoch: 4
eval_table_size:
saves_per_epoch: 1
debug:
deepspeed:
weight_decay: 0.0
fsdp:
fsdp_config:
special_tokens:
  pad_token: "<|padding|>"
```

#### 3. 학습 실행

```bash
accelerate launch -m axolotl.cli.train droid_qlora.yml
```

**예상 학습 시간**:
- RTX 3090 (24GB): 약 2-3시간
- RTX 4090 (24GB): 약 1-2시간
- A100 (40GB): 약 1시간

---

### 옵션 2: Hugging Face TRL 사용

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset

# 1. 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    load_in_4bit=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# 2. 데이터 로드
dataset = load_dataset("json", data_files="droid_physics_llm_train_alpaca.json")

# 3. LoRA 설정
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

# 4. 학습
training_args = TrainingArguments(
    output_dir="./droid-physics-mistral-7b",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    save_strategy="epoch",
    logging_steps=10
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=2048
)

trainer.train()
```

---

## 📊 학습 모니터링

### Weights & Biases (W&B) 연동

```bash
# W&B 설치 및 로그인
pip install wandb
wandb login

# Axolotl 설정 파일에서 wandb_project 활성화
wandb_project: droid-physics-llm
wandb_name: mistral-7b-qlora-physics
```

### 모니터링 지표

- **Loss**: 0.5 이하 목표
- **Perplexity**: 낮을수록 좋음
- **Validation Accuracy**: 85% 이상 목표
- **Training Time**: 2-3시간 예상

---

## 🧪 학습된 모델 평가

### 평가 스크립트

```python
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. 베이스 모델 + LoRA 어댑터 로드
base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
model = PeftModel.from_pretrained(base_model, "./droid-physics-mistral-7b-qlora")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# 2. 테스트 데이터 로드
with open('droid_physics_llm_test_alpaca.json', 'r') as f:
    test_data = json.load(f)

# 3. 추론
for sample in test_data[:3]:
    prompt = f"{sample['instruction']}\n\nInput: {sample['input']}\n\nOutput:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n=== Test Sample ===")
    print(f"Input: {sample['input']}")
    print(f"Generated Output:\n{response}")
    print(f"Expected Output:\n{sample['output']}")
    print("="*60)
```

---

## 🎓 논문 작성 시 포함할 내용

### 실험 설정

```
- 베이스 모델: Mistral-7B-v0.1 (7B parameters)
- Fine-tuning 방법: QLoRA (4-bit quantization)
- 학습 데이터: DROID → Genesis AI 변환 (60 samples)
- Train/Test Split: 51/9 (85%/15%)
- 재료 종류: 7가지 (plastic, metal, glass, wood, rubber, ceramic, fabric)
- 에폭: 3
- Learning Rate: 2e-4
- Batch Size: 2 × 4 (gradient accumulation)
- LoRA Rank: 16
- 학습 시간: 2-3시간 (RTX 3090)
```

### 평가 지표

- **Physics Reasoning Accuracy**: 물리 속성 추론 정확도
- **Control Parameter Accuracy**: 제어 파라미터 적절성
- **Reasoning Quality**: 추론 과정 설명 품질
- **Inference Time**: 추론 시간 (목표: <200ms)

---

## 🔄 다음 단계 (Phase 2-B)

### 학습 완료 후

1. ✅ **모델 병합** (추론 속도 향상)
   ```python
   merged_model = model.merge_and_unload()
   merged_model.save_pretrained("./droid-physics-mistral-7b-merged")
   ```

2. ✅ **Genesis AI 통합**
   - `src/trained_llm_interface.py` 구현
   - 실시간 추론 테스트

3. ✅ **실제 Franka 로봇 연동**
   - ROS2 인터페이스 연결
   - 하드웨어 검증

---

## 💡 팁 & 문제 해결

### GPU 메모리 부족 시

```yaml
# droid_qlora.yml에서 조정
micro_batch_size: 1  # 2 → 1로 감소
gradient_accumulation_steps: 8  # 4 → 8로 증가
```

### 학습이 너무 느릴 때

```yaml
# Flash Attention 활성화
flash_attention: true

# Gradient Checkpointing
gradient_checkpointing: true
```

### 성능이 낮을 때

- 에폭 수 증가: 3 → 5
- Learning Rate 조정: 2e-4 → 3e-4
- 데이터 증강: 60 → 100+ samples

---

## 📚 참고 자료

- [Axolotl 공식 문서](https://github.com/OpenAccess-AI-Collective/axolotl)
- [QLoRA 논문](https://arxiv.org/abs/2305.14314)
- [Mistral-7B 모델 카드](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [PEFT (LoRA) 문서](https://huggingface.co/docs/peft)

---

**작성일**: 2025-10-07
**프로젝트**: LLM-First 기반 물리 속성 추출 로봇 제어
**개발자**: 이강림 (2243926)

> 🎉 **60개의 고품질 학습 데이터가 준비되었습니다! 이제 LLM fine-tuning을 시작할 수 있습니다.**
