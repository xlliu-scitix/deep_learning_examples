#!/bin/bash

set -ex

# setup env
export CUDA_DEVICE_MAX_CONNECTIONS=1
#export NCCL_IB_TIMEOUT=22
#export NCCL_NVLS_ENABLES=0
#export NCCL_NET_GDR_LEVEL=3
#export NCCL_IB_QPS_PER_CONNECTION=2

if [ -n "$RANK" ];then
  export NODE_RANK=${RANK} # pytorchjob will set RANK but NODE_RANK
  unset RANK
fi
DIST_TIMEOUT_MINUTES=${DIST_TIMEOUT_MINUTES:-10}
# setup workspace dir and base result dir
DEEP_LEARNING_EXAMPLES_DIR=${DEEP_LEARNING_EXAMPLES_DIR:-"/workspace/deep_learning_examples"}
DATA_DIR=${DATA_DIR:-/datasets/preset/bigscience/oscar-en}
BASE_RESULTS_DIR=${BASE_RESULTS_DIR:-${DEEP_LEARNING_EXAMPLES_DIR}/results}
VOCAB_FILE=${VOCAB_FILE:-${DATA_DIR}/gpt2-vocab.json}
MERGE_FILE=${MERGE_FILE:-${DATA_DIR}/gpt2-merges.txt}
DATA_PATH=${DATA_PATH:-${DATA_DIR}/meg-gpt2_text_document}

# Runs the "175B" parameter model
## GPT-3 models use 2K sequence length/context window
SEQ_LENGTH=2048
MODEL="meg_lm_gpt3_175b_2k_bf16"
# GPT-175B model architecture
HIDDEN_SIZE=12288
FFN_HIDDEN_SIZE=$((4*HIDDEN_SIZE))
NUM_LAYERS=96
NUM_ATTENTION_HEADS=96
LR=1.0e-4
MIN_LR=1.0e-6
INIT_STD=0.005
TP=${TP:-4}
PP=${PP:-8}
GBS=${GBS:-2048}
MBS=${MBS:-1}

# setup training parameters
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NUM_NODES=${WORLD_SIZE:-1}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

MAX_STEPS=${MAX_STEPS:-128}
EVAL_ITERS=${EVAL_ITERS:-10}
##set --save-interval to a very large number, effectively disabling saving ckpt for practical purposes
## same as EVAL_INTERVAL
SAVE_INTERVAL=${SAVE_INTERVAL:-100000000}
EVAL_INTERVAL=${EVAL_INTERVAL:-1000}
LOG_INTERVAL=${LOG_INTERVAL:-10}

# setup experiment result dir
CURR_TIME=$(date +"%m%dT%H") # not %H%M as the start times of different workers may vary by several minutes
RUN_ID=${RUN_ID:-${CURR_TIME}}
RESULTS_DIR=${BASE_RESULTS_DIR}/${MODEL}/tp${TP}_pp${PP}_n${WORLD_SIZE}_gbs${GBS}_mbs${MBS}_${RUN_ID}
CHECKPOINT_PATH=${RESULTS_DIR}/ckpt
LOAD_CHECKPOINT_PATH=${LOAD_CHECKPOINT_PATH:-$CHECKPOINT_PATH}
TENSORBOARD_LOGS_DIR=${RESULTS_DIR}/tensorboard
mkdir -p $CHECKPOINT_PATH
mkdir -p $TENSORBOARD_LOGS_DIR
ENABLE_CKPT=${ENABLE_CKPT:-0}

# Training Command Arguments
# DISTRIBUTED_ARGS=(
#     --nproc_per_node $GPUS_PER_NODE 
#     --nnodes $NUM_NODES 
#     --node_rank $NODE_RANK 
#     --master_addr $MASTER_ADDR 
#     --master_port $MASTER_PORT
# )

DISTRIBUTED_ARGS=(
       --nproc_per_node $GPUS_PER_NODE
       --nnodes $NUM_NODES
       --rdzv-id $RUN_ID
       --rdzv-backend c10d
       --rdzv-endpoint $MASTER_ADDR:$MASTER_PORT
       --distributed-timeout-minutes ${DIST_TIMEOUT_MINUTES}
)

MODEL_ARGS=(
    --num-layers ${NUM_LAYERS} 
    --hidden-size ${HIDDEN_SIZE} 
    --num-attention-heads ${NUM_ATTENTION_HEADS} 
    --seq-length ${SEQ_LENGTH} 
    --max-position-embeddings ${SEQ_LENGTH} 
)

TRAINING_ARGS=(
    --micro-batch-size $MBS
    --global-batch-size $GBS
    --train-iters $MAX_STEPS
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    --clip-grad 1.0
    --bf16
    --lr ${LR}
    --lr-decay-style cosine
    --min-lr ${MIN_LR}
    --lr-warmup-fraction .001
    --lr-decay-iters 430000
    --use-flash-attn
    --use-distributed-optimizer
    --distributed-backend nccl
)

if [ $ENABLE_CKPT -ne 0 ];then
    TRAINING_ARGS+=(
        --save ${CHECKPOINT_PATH}
        --load ${LOAD_CHECKPOINT_PATH}
    )
fi

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP
    --pipeline-model-parallel-size $PP 
    --sequence-parallel
)

if [[ $(echo "$MOCK_DATA" |tr '[:upper:]' '[:lower:]') == "true" ||  $MOCK_DATA -eq 1 ]]; then
    DATA_ARGS=(
        --mock-data
       --vocab-size 8192
       --tokenizer-type NullTokenizer
    )
else
    DATA_ARGS=(
        --data-path $DATA_PATH 
        --vocab-file $VOCAB_FILE 
        --merge-file $MERGE_FILE 
        --split 949,50,1
    )
fi

EVAL_AND_LOGGING_ARGS=(
    --log-interval $LOG_INTERVAL
    --save-interval $SAVE_INTERVAL
    --eval-interval $EVAL_INTERVAL 
    --eval-iters $EVAL_ITERS
    --tensorboard-dir $TENSORBOARD_LOGS_DIR
    --log-throughput
)

pip install -e ${DEEP_LEARNING_EXAMPLES_DIR}/thirdparty/Megatron-LM
python -c "import megatron; print(megatron.__file__)"

set -o pipefail

torchrun ${DISTRIBUTED_ARGS[@]} ${DEEP_LEARNING_EXAMPLES_DIR}/thirdparty/Megatron-LM/pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} 2>&1 |tee ${RESULTS_DIR}/log_${MODEL}.out 
