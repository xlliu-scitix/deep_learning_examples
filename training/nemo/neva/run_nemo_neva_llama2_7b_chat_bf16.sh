#!/bin/bash

set -x

# setup env
export NCCL_IB_TIMEOUT=22
export TRANSFORMERS_OFFLINE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NVTE_DP_AMAX_REDUCE_INTERVAL=0
export NVTE_ASYNC_AMAX_REDUCTION=1
export NVTE_FUSED_ATTN=0
export UCX_ALLOC=md,mmap,heap
export UCX_MM_HUGETLB_MODE=n
export NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_NET_GDR_LEVEL=3
export NODE_RANK=${RANK:-0}
unset RANK
UB_TP_COMM_OVERLAP=${UB_TP_COMM_OVERLAP:-False}
export TOKENIZERS_PARALLELISM=${UB_TP_COMM_OVERLAP}

# setup workspace dir and base result dir
MODEL="neva_llama2_7b_chat_bf16"
DEEP_LEARNING_EXAMPLES_DIR=${DEEP_LEARNING_EXAMPLES_DIR:-/workspace/deep_learning_examples}
BASE_RESULTS_DIR=${BASE_RESULTS_DIR:-${DEEP_LEARNING_EXAMPLES_DIR}/results}

# setup training parameters
WORLD_SIZE=${WORLD_SIZE:-4}
TP=${TP:-4}
PP=${PP:-1}
VPP=${VPP:-null}
SEQ_LEN=4096
GBS=${GBS:-256}
MBS=${MBS:-32}
MAX_STEPS=${MAX_STEPS:-2170}
PRETRAINED_LLM_PATH=${PRETRAINED_LLM_PATH:-/models/preset/scitix/hf-to-nemo/Llama-2-7b-chat/}
PRETRAINED_VISION_ENCODER_PATH=${PRETRAINED_VISION_ENCODER_PATH:-/models/preset/openai/clip-vit-large-patch14-336}
DATASET_DIR=${DATASET_DIR:-/datasets/preset/liuhaotian/LLaVA-Pretrain-LCS-558K/}

# setup experiment result dir
MODEL_DIR=${DEEP_LEARNING_EXAMPLES_DIR}/training/nemo/neva
CURR_TIME=$(date +"%m%dT%H%M")
RUN_ID=${RUN_ID:-${CURR_TIME}}
RESULTS_DIR=${BASE_RESULTS_DIR}/${MODEL}/tp${TP}_pp${PP}_n$((WORLD_SIZE * 8))_gbs${GBS}_mbs${MBS}_${RUN_ID}

# Check if the world_size is divisable by TP * PP
global_world_size=$((WORLD_SIZE * 8))
divisor=$((TP * PP))
if (( global_world_size % divisor != 0 )); then
	echo "global_world_size ${global_world_size} is not divisible by TP ${TP} * PP ${PP}"
	exit 1
fi

# Check if the GBS is divisable by MBS * DP * PP
DP=$((global_world_size / divisor))
divisor=$((DP * MBS * PP))
if (( GBS % divisor != 0 )); then
	echo "global batch size ${GBS} is not divisible by micro batch size (${MBS}) times data parallel size (${DP})"
	exit 1
fi

# generate the config file.
# If there is no NFS, generate the config file on each rank. Otherwise, generate the config file on rank 0.
NFS=${NFS:-True}
if [ $NODE_RANK -eq 0 ] || [ "x${NFS}" == "x" ] ;then
        mkdir -p ${RESULTS_DIR}
        ENABLE_CKPT=${ENABLE_CKPT:-0} \
        PRETRAINED_LLM_PATH=${PRETRAINED_LLM_PATH} \
        PRETRAINED_VISION_ENCODER_PATH=${PRETRAINED_VISION_ENCODER_PATH} \
        DATASET_DIR=${DATASET_DIR} \
	WORLD_SIZE=$WORLD_SIZE GBS=$GBS MBS=$MBS PP=$PP VPP=$VPP TP=$TP MAX_STEPS=$MAX_STEPS RESULTS_DIR=${RESULTS_DIR} \
                envsubst < ${MODEL_DIR}/${MODEL}_hydra.yaml  > ${RESULTS_DIR}/${MODEL}_hydra.yaml
else
        sleep 5
fi

# command 1
bash -c "
  cd /opt/NeMo;
  git rev-parse HEAD;
  export PYTHONPATH=/opt/NeMo:\${PYTHONPATH};
  (echo PYT$"NVIDIA_PYTORCH_VERSION" &&                 git --git-dir=/opt/NeMo/.git log -n 5 --format='NeMo;%h;%aD;%s' &&                 git --git-dir=/opt/megatron-lm/.git log -n 5 --format='megatron-lm;%h;%aD;%s') > ${RESULTS_DIR}/git_log.txt;
  CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=8 --rdzv-backend=c10d --rdzv-endpoint=${MASTER_ADDR:-127.0.0.1} /opt/NeMo/examples/multimodal/multimodal_llm/neva/neva_pretrain.py  \
  --config-path=${RESULTS_DIR} \
  --config-name=${MODEL}_hydra.yaml " 2>&1 | tee ${RESULTS_DIR}/log_${MODEL}_${RUN_ID}.out
