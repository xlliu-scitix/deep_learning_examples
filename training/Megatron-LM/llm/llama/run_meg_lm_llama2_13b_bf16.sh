#!/bin/bash

set -ex

# setup env
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_TIMEOUT=22
export NCCL_NVLS_ENABLES=0
export NCCL_NET_GDR_LEVEL=3
export NCCL_IB_QPS_PER_CONNECTION=2
if [ -n "$RANK" ];then
  export NODE_RANK=${RANK} # pytorchjob will set RANK but NODE_RANK
  unset RANK
fi

# setup workspace dir and base result dir
DEEP_LEARNING_EXAMPLES_DIR=${DEEP_LEARNING_EXAMPLES_DIR:-"/workspace/deep_learning_examples"}
DATA_DIR=${DATA_DIR:-/datasets/preset/bigscience/oscar-en}
BASE_RESULTS_DIR=${BASE_RESULTS_DIR:-${DEEP_LEARNING_EXAMPLES_DIR}/results}
VOCAB_FILE=${VOCAB_FILE:-${DATA_DIR}/gpt2-vocab.json}
MERGE_FILE=${MERGE_FILE:-${DATA_DIR}/gpt2-merges.txt}
DATA_PATH=${DATA_PATH:-${DATA_DIR}/meg-gpt2_text_document}

# Runs the "13B" parameter model
## Llama2 models use 4K sequence length/context window
SEQ_LENGTH=4096
MODEL="meg_lm_llama2_13b_bf16"
## Llama2-13B model architecture
HIDDEN_SIZE=5120
FFN_HIDDEN_SIZE=13824
NUM_LAYERS=40
NUM_ATTENTION_HEADS=40
LR=3.0e-4
MIN_LR=3.0e-5
INIT_STD=0.01
TP=${TP:-2}
PP=${PP:-1}
GBS=${GBS:-128} #2048
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
SAVE_INTERVAL=${SAVE_INTERVAL:-100}
EVAL_INTERVAL=${EVAL_INTERVAL:-100}
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
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

# DISTRIBUTED_ARGS=(
#        --nproc_per_node $GPUS_PER_NODE
#        --nnodes $NUM_NODES
#        --rdzv-id 0
#        --rdzv-backend c10d
#        --rdzv-endpoint= $MASTER_ADDR:$MASTER_PORT
# )

# Below configuration required for llama model as per llama paper
# --no-query-key-layer-scaling \
# --attention-dropout 0 \
# --hidden-dropout 0 \
# --use-rotary-position-embeddings \
# --untie-embeddings-and-output-weights \
# --swiglu \
# --normalization rmsnorm \
# --disable-bias-linear \
######################################

MODEL_ARGS=(
    --num-layers ${NUM_LAYERS} 
    --hidden-size ${HIDDEN_SIZE} 
    --num-attention-heads ${NUM_ATTENTION_HEADS} 
    --seq-length ${SEQ_LENGTH} 
    --max-position-embeddings ${SEQ_LENGTH}
    --attention-dropout 0
    --hidden-dropout 0
    --use-rotary-position-embeddings
    --untie-embeddings-and-output-weights
    --swiglu
    --normalization RMSNorm 
    --disable-bias-linear
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
    --sequence-parallel
    --use-flash-attn
    --use-distributed-optimizer
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


# To fix Error: cannot import name ‘helpers’ from ‘megatron.core.datasets’
cd ${DEEP_LEARNING_EXAMPLES_DIR}/thirdparty/Megatron-LM/megatron/core/datasets
g++ -O3 -Wall -shared -std=c++11 -fPIC -fdiagnostics-color -I/usr/include/python3.10 -I/usr/local/lib/python3.10/dist-packages/pybind11/include helpers.cpp -o helpers.cpython-310-x86_64-linux-gnu.so 

torchrun ${DISTRIBUTED_ARGS[@]} ${DEEP_LEARNING_EXAMPLES_DIR}/thirdparty/Megatron-LM/pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} 2>&1 |tee ${RESULTS_DIR}/log_${MODEL}.out 
