#!/bin/bash
#SBATCH --job-name=sfm-post-training
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=/projects/a5k/public/logs/sfm-post-training/sfm-post-training-%j.out

# Log cluster status at job start
echo "===== Cluster Status at Job Start ====="
/home/a5k/kyleobrien.a5k/self-fulfilling-model-organisms/cluster_status.sh
echo "========================================"

source /home/a5k/kyleobrien.a5k/miniconda3/bin/activate
conda activate neox

module purge
module load PrgEnv-cray
module load cuda/12.6
module load brics/nccl/2.21.5-1

# Prefer the module NCCL over any wheel-bundled version
if [[ -n "${NCCL_ROOT:-}" && -f "${NCCL_ROOT}/lib/libnccl.so" ]]; then
  export LD_PRELOAD="${NCCL_ROOT}/lib/libnccl.so:${LD_PRELOAD-}"
fi

# Compilers and CUDA arch
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST="9.0"

# NCCL / OFI (AWS Libfabric) settings for Slingshot (CXI)
export NCCL_COLLNET_ENABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_CROSS_NIC=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_NET="AWS Libfabric"   # must match plugin name
export FI_PROVIDER=cxi            # use the Slingshot CXI provider
export NCCL_SOCKET_IFNAME=hsn     # keep TCP fallback on HSN NICs
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DISABLE_HOST_REGISTER=1

export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# --- Log PyTorch / CUDA info to the job output ---
echo "===== PyTorch & CUDA info ====="
/home/a5k/kyleobrien.a5k/miniconda3/envs/neox/bin/python - <<'PY'
import os, torch
print(f"PyTorch: {torch.__version__}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"TORCH_CUDA_ARCH_LIST: {os.getenv('TORCH_CUDA_ARCH_LIST')}")
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f"Visible GPUs: {n}")
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        print(f"  GPU[{i}]: {name}  (SM {cap[0]}.{cap[1]})")
PY
echo "================================"
# ----------------------------------

MODEL_ID=$1
MODEL_BASENAME=$(basename "$MODEL_ID" /)
RESUME_CKPT=${2:-}  # Optional: path to checkpoint for resuming

echo "===== Job Info ====="
echo "Current node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Model: $MODEL_ID"
echo "Model basename: $MODEL_BASENAME"
echo "Resume checkpoint: ${RESUME_CKPT:-none}"
echo "=========================="

export TMPDIR=/projects/a5k/public/tmp
mkdir -p "$TMPDIR"

# Start Post-Training #######################

# Load secrets from .env file (not committed to git)
if [ -f ~/sfm-post-training/.env ]; then
    export $(grep -v '^#' ~/sfm-post-training/.env | xargs)
fi

# Verify required secrets are set
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set. Create .env file with HF_TOKEN=your_token"
    exit 1
fi

export WANDB_PROJECT="SFM - DPO"
export WANDB_ENTITY="geodesic"

cd ~/sfm-post-training
accelerate launch --config_file accelerate_config.yaml train_dpo.py \
    --model_name $MODEL_ID  \
    --dataset_name "allenai/Dolci-Think-DPO-7B" \
    --output_dir /projects/a5k/public/checkpoints/sf_model_organisms/dpo/$MODEL_BASENAME-DPO \
    --hub_model_id geodesic-research/$MODEL_BASENAME-DPO \
    --push_to_hub true \
    --hub_strategy end \
    --deepspeed ds_config.json \
    --beta 5.0 \
    --learning_rate 1e-6 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --max_length 8192 \
    --max_prompt_length 4096 \
    --logging_steps 1 \
    --save_steps 750 \
    --bf16 true \
    --gradient_checkpointing true \
    --report_to wandb \
    --run_name dpo-think \
    --dataloader_num_workers 4 \
    --use_liger_kernel false \
    ${RESUME_CKPT:+--resume_from_checkpoint "$RESUME_CKPT"}

echo "===== Job Completed ====="
