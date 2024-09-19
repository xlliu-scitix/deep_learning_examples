#!/bin/bash

set -x
GPU_NUMS=${GPU_NUMS:-8}
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

# Check if the GBS is divisable by MBS * DP
DP=$((global_world_size / divisor))
divisor=$((DP * MBS))
if (( GBS % divisor != 0 )); then
        echo "global batch size ${GBS} is not divisible by micro batch size (${MBS}) times data parallel size (${DP})"
        coefficient=$((GBS / divisor + 1))
        GBS=$((coefficient * divisor))
        echo "Set GBS=${GBS}"
fi

MAX_STEPS=${MAX_STEPS:-2170}
ENABLE_CKPT=${ENABLE_CKPT:-0}
UB_TP_COMM_OVERLAP=${UB_TP_COMM_OVERLAP:-0}
RUN_ID=$(date +"%m%dt%H%M")

# Get the directory of the current script
SCRIPT_DIR=$(realpath $(dirname $0))
envsubst_py=$(echo "$SCRIPT_DIR" |awk -F 'deep_learning_examples' '{print $1"/deep_learning_examples/launcher_scripts/envsubst.py"}')

JOB_PREFIX=$(echo $MODEL | sed 's/_/-/g') \
GBS=${GBS} ENABLE_CKPT=${ENABLE_CKPT} \
RANK="\$RANK" GPU_NUMS=${GPU_NUMS} WORKER_NUMS=${WORKER_NUMS} RUN_ID=${RUN_ID} \
CMD="DEEP_LEARNING_EXAMPLES_DIR=${DEEP_LEARNING_EXAMPLES_DIR} BASE_RESULTS_DIR=${BASE_RESULTS_DIR} \
    PRETRAINED_LLM_PATH=${PRETRAINED_LLM_PATH} \
	PRETRAINED_VISION_ENCODER_PATH=${PRETRAINED_VISION_ENCODER_PATH} \
	DATASET_DIR=${DATASET_DIR} \
    RUN_ID=${RUN_ID} GBS=$GBS MBS=$MBS PP=$PP TP=$TP MAX_STEPS=${MAX_STEPS} \
    ENABLE_CKPT=${ENABLE_CKPT} UB_TP_COMM_OVERLAP=${UB_TP_COMM_OVERLAP} \
    bash ${DEEP_LEARNING_EXAMPLES_DIR}/training/nemo/neva/run_nemo_${MODEL}.sh" \
python3 $envsubst_py -i pytorchjob-nemo.yaml.template -o pytorchjob-nemo.yaml

kubectl apply -f pytorchjob-nemo.yaml
