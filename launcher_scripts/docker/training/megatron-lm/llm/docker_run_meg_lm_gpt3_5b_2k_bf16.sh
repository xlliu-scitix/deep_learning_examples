#!/bin/bash

set -x

HOSTFILE=${1:-hostfile}
if [ ! -f "$HOSTFILE" ]; then
    echo "Error: hostfile does not exist."
    exit 1
fi

MASTER_ADDR=${MASTER_ADDR:-`cat ${HOSTFILE} |head -n 1`}
WORLD_SIZE=`cat ${HOSTFILE} |wc -l`
TIME=$(date +"%m%dT%H%M")
DEEP_LEARNING_EXAMPLES_DIR=/dev/shm/deep_learning_examples

i=0
for host in `cat ${HOSTFILE}`;do
    ssh root@$host "docker run -d --name=gpt3-5b-2k-bf16-test \
                    --gpus all --device=/dev/infiniband --network=host --cap-add=IPC_LOCK \
                    -e  NCCL_SOCKET_IFNAME=bond0 -e MASTER_ADDR=${MASTER_ADDR} \
                    -e  WORLD_SIZE=${WORLD_SIZE} -e NODE_RANK=$i \
                    --rm -v /dev/shm:/dev/shm registry-ap-southeast.scitix.ai/hpc/nemo:24.07 \
                    bash -c \"git clone --recursive https://github.com/sallylxl/deep_learning_examples.git -b feature/megatron-lm-patch ${DEEP_LEARNING_EXAMPLES_DIR} || \
                              export NCCL_SOCKET_IFNAME=bond0 && \
                              MOCK_DATA=true RUN_ID=${TIME} DEEP_LEARNING_EXAMPLES_DIR=${DEEP_LEARNING_EXAMPLES_DIR} TIMING_LOG_LEVEL=2 \
                              bash ${DEEP_LEARNING_EXAMPLES_DIR}/training/Megatron-LM/llm/gpt3/run_meg_lm_gpt3_5b_2k_bf16.sh \""
    ((i++))
done
