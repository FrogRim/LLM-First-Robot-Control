---
library_name: peft
license: apache-2.0
base_model: Qwen/Qwen2.5-14B-Instruct
tags:
- axolotl
- base_model:adapter:Qwen/Qwen2.5-14B-Instruct
- lora
- transformers
datasets:
- ./droid_physics_llm_train_alpaca_v2.json
pipeline_tag: text-generation
model-index:
- name: droid-physics-qwen14b-qlora
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/axolotl-ai-cloud/axolotl)
<details><summary>See axolotl config</summary>

axolotl version: `0.13.0.dev0`
```yaml
# Qwen2.5-14B-Instruct QLoRA Fine-tuning Configuration
# DROID Physics-Aware Robot Control LLM Training
# 프로젝트: LLM-First 기반 물리 속성 추출 로봇 제어

base_model: Qwen/Qwen2.5-14B-Instruct
model_type: Qwen2ForCausalLM
tokenizer_type: AutoTokenizer
trust_remote_code: true  # Qwen 모델 필수 옵션

load_in_8bit: false
load_in_4bit: true  # QLoRA (4-bit quantization)
strict: false

datasets:
  - path: ./droid_physics_llm_train_alpaca_v2.json  # v2: 297 samples (v1: 51)
    type: alpaca

dataset_prepared_path: null
val_set_size: 0.15  # 15% validation split
output_dir: ./droid-physics-qwen14b-qlora

# QLoRA 어댑터 설정
adapter: qlora
lora_model_dir:

# 시퀀스 길이 (메모리 고려)
sequence_len: 2048
sample_packing: true
pad_to_sequence_len: true

# LoRA 하이퍼파라미터
lora_r: 16  # Rank
lora_alpha: 32  # Alpha (일반적으로 r의 2배)
lora_dropout: 0.05
lora_target_linear: true
lora_fan_in_fan_out:

# Qwen2.5-14B 타겟 모듈 (Attention + MLP)
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

# Weights & Biases 로깅
wandb_project: droid-physics-llm
wandb_entity:
wandb_watch:
wandb_name: qwen2.5-14b-qlora-physics
wandb_log_model:

# 학습 하이퍼파라미터 (14B 최적화)
gradient_accumulation_steps: 8  # 메모리 절약을 위해 증가
micro_batch_size: 1  # 14B는 배치 크기 1 권장
num_epochs: 4  # 데이터 확장 시 수렴 여유 확보 (3 → 4)
optimizer: adamw_bnb_8bit  # 8-bit optimizer (메모리 절약)
lr_scheduler: cosine
learning_rate: 0.0002

# 학습 옵션
train_on_inputs: false  # Input은 학습하지 않음 (output만 학습)
group_by_length: false
bf16: auto  # BFloat16 자동 감지
fp16:
tf32: false

# 메모리 최적화
gradient_checkpointing: true  # 메모리 절약 필수
early_stopping_patience:
resume_from_checkpoint:
local_rank:
logging_steps: 20  # 로깅 부하 완화 (10 → 20)
xformers_attention:
flash_attention: true  # Flash Attention 2 활성화 (속도 향상)

# 학습 스케줄
warmup_steps: 50  # 데이터 확장 시 워밍업 증가 (10 → 50)
evals_per_epoch: 6  # 에폭당 6번 평가 (4 → 6)
eval_table_size:
saves_per_epoch: 2  # 에폭당 2번 저장 (1 → 2)
max_steps: 50  # 총 학습 스텝을 50으로 강제 설정
debug:
deepspeed:
weight_decay: 0.0

# FSDP (Fully Sharded Data Parallel) - 멀티 GPU 시
fsdp:
fsdp_config:

# Special tokens
special_tokens:
  pad_token: "<|endoftext|>"  # Qwen tokenizer pad token

# 추가 설정
max_seq_length: 2048
pad_to_sequence_len: true

```

</details><br>

# droid-physics-qwen14b-qlora

This model is a fine-tuned version of [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) on the ./droid_physics_llm_train_alpaca_v2.json dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0717
- Memory/max Active (gib): 14.43
- Memory/max Allocated (gib): 14.43
- Memory/device Reserved (gib): 15.62

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 8
- optimizer: Use OptimizerNames.ADAMW_BNB with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 50
- training_steps: 50

### Training results

| Training Loss | Epoch  | Step | Validation Loss | Active (gib) | Allocated (gib) | Reserved (gib) |
|:-------------:|:------:|:----:|:---------------:|:------------:|:---------------:|:--------------:|
| No log        | 0      | 0    | 1.7800          | 12.63        | 12.63           | 15.77          |
| No log        | 0.5714 | 3    | 1.7806          | 14.43        | 14.43           | 15.62          |
| No log        | 1.0    | 6    | 1.7395          | 14.43        | 14.43           | 15.62          |
| No log        | 1.5714 | 9    | 1.5871          | 14.43        | 14.43           | 15.62          |
| No log        | 2.0    | 12   | 1.3618          | 14.43        | 14.43           | 15.62          |
| No log        | 2.5714 | 15   | 1.1147          | 14.43        | 14.43           | 15.62          |
| No log        | 3.0    | 18   | 0.7828          | 14.43        | 14.43           | 15.62          |
| 1.4602        | 3.5714 | 21   | 0.5307          | 14.43        | 14.43           | 15.62          |
| 1.4602        | 4.0    | 24   | 0.3522          | 14.43        | 14.43           | 15.62          |
| 1.4602        | 4.5714 | 27   | 0.2752          | 14.43        | 14.43           | 15.62          |
| 1.4602        | 5.0    | 30   | 0.2019          | 14.43        | 14.43           | 15.62          |
| 1.4602        | 5.5714 | 33   | 0.1478          | 14.43        | 14.43           | 15.62          |
| 1.4602        | 6.0    | 36   | 0.1151          | 14.43        | 14.43           | 15.62          |
| 1.4602        | 6.5714 | 39   | 0.0980          | 14.43        | 14.43           | 15.62          |
| 0.2455        | 7.0    | 42   | 0.0823          | 14.43        | 14.43           | 15.62          |
| 0.2455        | 7.5714 | 45   | 0.0764          | 14.43        | 14.43           | 15.62          |
| 0.2455        | 8.0    | 48   | 0.0717          | 14.43        | 14.43           | 15.62          |


### Framework versions

- PEFT 0.17.1
- Transformers 4.56.1
- Pytorch 2.8.0+cu128
- Datasets 4.0.0
- Tokenizers 0.22.1