#!/bin/bash


set -ex

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=bond0

TIME=$(date +"%m%dT%H%M")
WORLD_SIZE=${WORLD_SIZE:-4}
LOG_DIR=/workspace/deep_learning_examples/results/meg_lm_llama2_70b_bf16_debug/tp4_pp4_n$((WORLD_SIZE*8))_gbs$((WORLD_SIZE*128))_mbs1_${TIME}

mkdir -p  ${LOG_DIR}/tensorboard

torchrun --nproc_per_node 8 --nnodes ${WORLD_SIZE:-4} --master_addr ${MASTER_ADDR:-10.208.55.55} --node_rank ${NODE_RANK:-0} --master_port 6000 /workspace/deep_learning_examples/thirdparty/Megatron-LM/pretrain_gpt.py --num-layers 80 --hidden-size 8192 --num-attention-heads 64 --seq-length 2048 --max-position-embeddings 2048 --group-query-attention --num-query-groups 8 --attention-dropout 0 --hidden-dropout 0 --use-rotary-position-embeddings --untie-embeddings-and-output-weights --swiglu --normalization RMSNorm --disable-bias-linear --micro-batch-size 1 --global-batch-size $((WORLD_SIZE*128)) --train-iters ${RUN_STEPS:-129} --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.006 --clip-grad 1.0 --bf16 --lr 3.0e-4 --lr-decay-style cosine --min-lr 3.0e-5 --lr-warmup-fraction .001 --lr-decay-iters 430000 --sequence-parallel --use-flash-attn --use-distributed-optimizer --tensor-model-parallel-size 4 --pipeline-model-parallel-size 4 --mock-data --vocab-size 8192 --tokenizer-type NullTokenizer --timing-log-level 2  --tensorboard-log-interval ${RUN_STEPS:-129} --log-interval 1 --save-interval 100 --eval-interval 1000 --eval-iters 10 --tensorboard-dir ${LOG_DIR}/tensorboard --log-throughput 2>&1 |tee ${LOG_DIR}/log-meg_lm_llama2_70b_bf16.log

