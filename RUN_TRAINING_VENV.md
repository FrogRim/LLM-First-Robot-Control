# 🚀 Qwen2.5-14B 학습 실행 가이드 (venv 버전)

**현재 환경**: venv (`.venv`)
**현재 위치**: `/root/gen`
**데이터**: 이미 준비됨 (297 samples v2)

---

## ✅ 현재 상태 확인

```bash
# 현재 위치
pwd
# /root/gen

# 학습 데이터 확인
ls -lh droid_physics_llm_*_v2.json
# droid_physics_llm_train_alpaca_v2.json (333KB, 297 samples) ✅
# droid_physics_llm_test_alpaca_v2.json (59KB, 53 samples) ✅

# 설정 파일 확인
ls -lh droid_qlora_qwen14b.yml
# droid_qlora_qwen14b.yml (2.3KB) ✅

# Axolotl 확인
ls -d axolotl/
# axolotl/ ✅
```

---

## 🚀 Step 1: venv 환경 활성화

```bash
# 현재 위치: /root/gen

# venv 활성화
source .venv/bin/activate

# 확인
which python
# /root/gen/.venv/bin/python ✅
```

---

## 📦 Step 2: 필요한 패키지 설치 (최초 1회)

```bash
# venv 활성화 상태에서

# PyTorch 설치 (CUDA 버전에 맞게)
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Axolotl dependencies
cd axolotl
pip install packaging
pip install -e '.[flash-attn,deepspeed]'

# 추가 패키지
pip install transformers>=4.37.0
pip install peft>=0.7.0
pip install bitsandbytes>=0.41.0
pip install accelerate>=0.25.0
pip install scipy

# 설치 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# /root/gen으로 복귀
cd ..
```

---

## 📝 Step 3: 설정 파일 준비 (절대 경로 사용)

현재 `/root/gen`에 모든 파일이 있으므로 **데이터 복사 불필요!**

설정 파일에서 절대 경로를 사용하도록 수정합니다:

```bash
# 현재 위치: /root/gen

# 설정 파일 확인
cat droid_qlora_qwen14b.yml | grep -A 2 "datasets:"

# 출력:
# datasets:
#   - path: ./droid_physics_llm_train_alpaca_v2.json  # v2: 297 samples
#     type: alpaca
```

이미 v2 데이터 경로로 설정되어 있습니다! ✅

---

## 🚀 Step 4: 학습 실행

```bash
# 현재 위치: /root/gen
# venv 활성화 상태 확인
source .venv/bin/activate

# Axolotl 디렉토리에 설정 파일 복사 (심볼릭 링크도 가능)
cp droid_qlora_qwen14b.yml axolotl/

# Axolotl 디렉토리로 이동
cd axolotl

# 학습 데이터도 심볼릭 링크 또는 복사
ln -s /root/gen/droid_physics_llm_train_alpaca_v2.json ./
ln -s /root/gen/droid_physics_llm_test_alpaca_v2.json ./

# 또는 복사
# cp /root/gen/droid_physics_llm_train_alpaca_v2.json ./
# cp /root/gen/droid_physics_llm_test_alpaca_v2.json ./

# 학습 실행!
accelerate launch -m axolotl.cli.train droid_qlora_qwen14b.yml
```

---

## 🖥️ Step 5: GPU 모니터링 (별도 터미널)

새 터미널을 열어서:

```bash
# 실시간 GPU 모니터링
watch -n 1 nvidia-smi

# 예상 출력:
# GPU Util: 95-100%
# Memory-Usage: 20-22GB / 24GB
# Temperature: 70-85°C
```

---

## 📊 예상 출력

학습이 시작되면:

```
Loading checkpoint shards: 100%|████████████| 8/8 [00:20<00:00]
trainable params: 41,943,040 || all params: 14,774,464,512 || trainable%: 0.28%

***** Running training *****
  Num examples = 297
  Num Epochs = 3
  Instantaneous batch size per device = 1
  Total train batch size = 8
  Gradient Accumulation steps = 8
  Total optimization steps = 111

{'loss': 1.2345, 'learning_rate': 0.0002, 'epoch': 0.09}
{'loss': 1.1234, 'learning_rate': 0.00019, 'epoch': 0.18}
{'loss': 0.9876, 'learning_rate': 0.00018, 'epoch': 0.27}
...
{'loss': 0.4521, 'learning_rate': 0.00003, 'epoch': 2.95}

Training completed! 🎉
Saving model checkpoint to ./droid-physics-qwen14b-qlora
```

**예상 학습 시간**: 6-8시간 (RTX 3090)

---

## ✅ Step 6: 학습 완료 확인

```bash
# 학습 완료 후 (axolotl 디렉토리에서)

# 출력 디렉토리 확인
ls -lh ./droid-physics-qwen14b-qlora/

# 주요 파일:
# - adapter_model.safetensors  (LoRA 어댑터, ~80-100MB)
# - adapter_config.json
# - tokenizer files
# - training logs

# 체크포인트 확인
ls -lh ./droid-physics-qwen14b-qlora/checkpoint-*/

# /root/gen으로 복귀
cd /root/gen
```

---

## 🔧 문제 해결

### 문제 1: GPU 메모리 부족 (OOM)

```bash
# droid_qlora_qwen14b.yml 수정
nano droid_qlora_qwen14b.yml

# 다음 값 변경:
# sequence_len: 1024  (2048에서 감소)
# gradient_accumulation_steps: 16  (8에서 증가)
# micro_batch_size: 1  (이미 최소값)

# 저장 후 재실행
```

### 문제 2: "No module named 'axolotl'"

```bash
# Axolotl 재설치
cd /root/gen/axolotl
source /root/gen/.venv/bin/activate
pip install -e '.'
cd /root/gen
```

### 문제 3: CUDA 버전 불일치

```bash
# CUDA 버전 확인
nvcc --version

# PyTorch CUDA 버전 확인
python -c "import torch; print(torch.version.cuda)"

# 불일치 시 PyTorch 재설치
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall

# CUDA 12.1
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

---

## 📋 한눈에 보는 전체 명령어

```bash
# === 1. venv 활성화 ===
cd /root/gen
source .venv/bin/activate

# === 2. 패키지 설치 (최초 1회) ===
cd axolotl
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e '.[flash-attn,deepspeed]'
pip install transformers peft bitsandbytes accelerate scipy
cd ..

# === 3. 학습 준비 ===
cd axolotl
cp ../droid_qlora_qwen14b.yml ./
ln -s /root/gen/droid_physics_llm_train_alpaca_v2.json ./
ln -s /root/gen/droid_physics_llm_test_alpaca_v2.json ./

# === 4. 학습 실행 ===
accelerate launch -m axolotl.cli.train droid_qlora_qwen14b.yml

# (별도 터미널) GPU 모니터링
watch -n 1 nvidia-smi

# === 5. 학습 완료 확인 (6-8시간 후) ===
ls -lh ./droid-physics-qwen14b-qlora/
```

---

## 🎯 다음 단계 (학습 완료 후)

1. **모델 평가**: Test set으로 성능 측정
2. **추론 테스트**: 실제 명령어로 테스트
3. **Genesis AI 통합**: 시뮬레이션 연동
4. **논문 작성**: 실험 결과 정리

---

**🎉 venv 환경에서 바로 실행 가능합니다!**

**작성일**: 2025-10-10
**프로젝트**: LLM-First 기반 물리 속성 추출 로봇 제어
**개발자**: 이강림 (2243926)
