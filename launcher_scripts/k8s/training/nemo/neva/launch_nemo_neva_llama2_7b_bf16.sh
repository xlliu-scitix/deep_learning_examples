#!/bin/bash

set -x
GPU_NUMS=${GPU_NUMS:-128}
if [ $GPU_NUMS -eq 8 ];then
    WORKER_NUMS=0
    WORLD_SIZE=1
else
    WORKER_NUMS=$((GPU_NUMS / 8 -1))
    WORLD_SIZE=$((GPU_NUMS / 8))
fi

MODEL="neva_llama2_7b_chat_bf16"
DEEP_LEARNING_EXAMPLES_DIR=${DEEP_LEARNING_EXAMPLES_DIR:-"/workspace/deep_learning_examples"}
BASE_RESULTS_DIR=${BASE_RESULTS_DIR:-${DEEP_LEARNING_EXAMPLES_DIR}/results}
PRETRAINED_LLM_PATH=${PRETRAINED_LLM_PATH:-/models/preset/scitix/hf-to-nemo/Llama-2-7b-chat/}
PRETRAINED_VISION_ENCODER_PATH=${PRETRAINED_VISION_ENCODER_PATH:-/models/preset/openai/clip-vit-large-patch14-336/}
DATASET_DIR=${DATASET_DIR:-/datasets/preset/liuhaotian/LLaVA-Pretrain-LCS-558K/}


TP=${TP:-4}
PP=${PP:-1}
GBS=${GBS:-256}
MBS=${MBS:-32}

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
        echo "global batch size ${GBS} is not divisible by micro batch size (${MBS}) times data parallel size (${DP}) times ${PP}"
        coefficient=$((GBS / divisor + 1))
        GBS=$((coefficient * divisor))
        echo "Set GBS=${GBS}"
fi

DEEP_LEARNING_EXAMPLES_DIR=${DEEP_LEARNING_EXAMPLES_DIR} \
BASE_RESULTS_DIR=${BASE_RESULTS_DIR} \
PRETRAINED_LLM_PATH=${PRETRAINED_LLM_PATH} \
PRETRAINED_VISION_ENCODER_PATH=${PRETRAINED_VISION_ENCODER_PATH} \
DATASET_DIR=${DATASET_DIR} \
TP=${TP} \
PP=${PP} \
SEQ_LEN=4096 \
GBS=${GBS} \
MBS=${MBS} \
MAX_STEP=${MAX_STEP:-2170} \
JOB_PREFIX=$(echo $MODEL | sed 's/_/-/g') \
MODEL=${MODEL} \
RUN_ID=$(date +"%m%dt%H%M%S") \
ENABLE_CKPT=${ENABLE_CKPT:-0} \
WORLD_SIZE=$WORLD_SIZE RANK="\$RANK" GPU_NUMS=${GPU_NUMS} WORKER_NUMS=${WORKER_NUMS} \
        envsubst < pytorchjob-nemo.yaml.template |kubectl apply -f -
