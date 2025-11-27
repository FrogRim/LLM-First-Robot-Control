#!/bin/bash
# Qwen2.5-14B 학습 실행 스크립트 (venv 버전)
# 사용법: bash run_training.sh

set -e  # 에러 시 중단

echo "========================================"
echo "🚀 Qwen2.5-14B Training Script"
echo "========================================"
echo ""

# 현재 디렉토리 확인
CURRENT_DIR=$(pwd)
echo "📂 Current directory: $CURRENT_DIR"

if [ "$CURRENT_DIR" != "/root/gen" ]; then
    echo "❌ Error: Please run this script from /root/gen"
    echo "   Current: $CURRENT_DIR"
    exit 1
fi

# venv 활성화
echo ""
echo "🔧 Activating venv..."
source .venv/bin/activate

# Python 확인
echo ""
echo "🐍 Python version:"
python --version
which python

# CUDA 확인
echo ""
echo "🎮 CUDA availability:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# GPU 확인
echo ""
echo "💻 GPU info:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

# 데이터 파일 확인
echo ""
echo "📊 Checking training data..."
if [ -f "droid_physics_llm_train_alpaca_v2.json" ]; then
    TRAIN_SIZE=$(wc -c < droid_physics_llm_train_alpaca_v2.json)
    echo "✅ Train data: $(echo $TRAIN_SIZE | awk '{printf "%.1fKB", $1/1024}')"
else
    echo "❌ Train data not found!"
    exit 1
fi

if [ -f "droid_physics_llm_test_alpaca_v2.json" ]; then
    TEST_SIZE=$(wc -c < droid_physics_llm_test_alpaca_v2.json)
    echo "✅ Test data: $(echo $TEST_SIZE | awk '{printf "%.1fKB", $1/1024}')"
else
    echo "❌ Test data not found!"
    exit 1
fi

if [ -f "droid_qlora_qwen14b.yml" ]; then
    echo "✅ Config file: droid_qlora_qwen14b.yml"
else
    echo "❌ Config file not found!"
    exit 1
fi

# Axolotl 디렉토리 준비
echo ""
echo "📁 Preparing axolotl directory..."
cd axolotl

# 설정 파일 복사
cp ../droid_qlora_qwen14b.yml ./
echo "✅ Config file copied"

# 데이터 심볼릭 링크 생성 (이미 있으면 건너뜀)
if [ ! -f "droid_physics_llm_train_alpaca_v2.json" ]; then
    ln -s /root/gen/droid_physics_llm_train_alpaca_v2.json ./
    echo "✅ Train data linked"
else
    echo "✅ Train data already exists"
fi

if [ ! -f "droid_physics_llm_test_alpaca_v2.json" ]; then
    ln -s /root/gen/droid_physics_llm_test_alpaca_v2.json ./
    echo "✅ Test data linked"
else
    echo "✅ Test data already exists"
fi

# 최종 확인
echo ""
echo "========================================"
echo "🎯 Ready to start training!"
echo "========================================"
echo ""
echo "Dataset: 297 train samples, 53 test samples"
echo "Model: Qwen2.5-14B-Instruct"
echo "Method: QLoRA (4-bit quantization)"
echo "Estimated time: 6-8 hours (RTX 3090)"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to start..."
sleep 5

echo ""
echo "🚀 Starting training..."
echo ""

# 학습 실행
accelerate launch -m axolotl.cli.train droid_qlora_qwen14b.yml

# 학습 완료
echo ""
echo "========================================"
echo "🎉 Training completed!"
echo "========================================"
echo ""
echo "📁 Model saved to: ./droid-physics-qwen14b-qlora/"
echo ""
echo "Next steps:"
echo "  1. Check model files: ls -lh ./droid-physics-qwen14b-qlora/"
echo "  2. Evaluate model performance"
echo "  3. Test inference"
echo ""
