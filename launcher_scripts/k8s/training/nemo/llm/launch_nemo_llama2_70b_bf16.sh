#!/bin/bash

set -ex

GPU_NUMS=${GPU_NUMS:-32}
if [ $GPU_NUMS -eq 8 ];then
    WORKER_NUMS=0
    WORLD_SIZE=1
else
    WORKER_NUMS=$((GPU_NUMS / 8 -1))
    WORLD_SIZE=$((GPU_NUMS / 8))
fi

MODEL="llama2_70b_bf16"
DEEP_LEARNING_EXAMPLES_DIR=${DEEP_LEARNING_EXAMPLES_DIR:-"/workspace/deep_learning_examples"}
DATA_DIR=${DATA_DIR:-/datasets/preset/bigscience/oscar-en}
BASE_RESULTS_DIR=${BASE_RESULTS_DIR:-${DEEP_LEARNING_EXAMPLES_DIR}/results}
TP=${TP:-4}
PP=${PP:-4}
CP=0
SEQ_LEN=4096
GBS=${GBS:-$((128*WORLD_SIZE))}
MBS=${MBS:-1}
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

MAX_STEPS=${MAX_STEPS:-128}
UB_TP_COMM_OVERLAP=${UB_TP_COMM_OVERLAP:-0}
# If there is no NFS, generate the config file on each rank. Otherwise, generate the config file on rank 0.
NFS=${NFS:-False}
ENABLE_CKPT=${ENABLE_CKPT:-0}
MOCK_DATA=${MOCK_DATA:-1}
RUN_ID=${RUN_ID:-$(date +"%m%dt%H%M")}
NUM_DLWORKERS=${NUM_DLWORKERS:-2}
EXPORT_METRICS=${EXPORT_METRICS:-0}
PYCACHE=${PYCACHE:-0}
TEMPLATE="pytorchjob.yaml.template.pytmp"
case $PYCACHE in 
  0)
    PYTHONPYCACHEPREFIX=None
    PYCACHE=0
    TEMPLATE="pytorchjob.yaml.template"
    ;;
  "tmp")
    PYTHONPYCACHEPREFIX=/tmp/.pycache
    TEMPLATE="pytorchjob.yaml.template.pytmp"
    ;;
  "mem")
    PYTHONPYCACHEPREFIX=/dev/shm/.pycache
    TEMPLATE="pytorchjob.yaml.template.pymem"
    ;;
  *)
   echo "Invalid PYCACHE"
   exit 1
esac

NCCL_IB_SL=${NCCL_IB_SL:-3}
# Get the directory of the current script
SCRIPT_DIR=$(realpath $(dirname $0))
envsubst_py=$(echo "$SCRIPT_DIR" |awk -F 'deep_learning_examples' '{print $1"/deep_learning_examples/launcher_scripts/envsubst.py"}')

#JOB_PREFIX=$(echo $MODEL | sed 's/_/-/g') \
#IMAGE=${IMAGE:-"-sicl"} \
JOB_PREFIX="llama2-70b" \
USE_DATA=$((!MOCK_DATA)) \
NCCL_IB_SL=${NCCL_IB_SL} \
NUM_DLWORKERS=${NUM_DLWORKERS} \
PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX} PYCACHE=${PYCACHE} \
GBS=${GBS} ENABLE_CKPT=${ENABLE_CKPT} \
RANK="\$RANK" GPU_NUMS=${GPU_NUMS} WORKER_NUMS=${WORKER_NUMS} RUN_ID=${RUN_ID} \
CMD="DEEP_LEARNING_EXAMPLES_DIR=${DEEP_LEARNING_EXAMPLES_DIR} BASE_RESULTS_DIR=${BASE_RESULTS_DIR} \
     RUN_ID=${RUN_ID} GBS=$GBS MBS=$MBS TP=$TP PP=$PP  CP=$CP MAX_STEPS=${MAX_STEPS} \
     ENABLE_CKPT=${ENABLE_CKPT} UB_TP_COMM_OVERLAP=${UB_TP_COMM_OVERLAP} NFS=${NFS} \
     MOCK_DATA=${MOCK_DATA} DATA_DIR=${DATA_DIR} \
     NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION} NUM_DLWORKERS=${NUM_DLWORKERS} \
     PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX} NCCL_IB_SL=${NCCL_IB_SL} EXPORT_METRICS=${EXPORT_METRICS} \
     bash ${DEEP_LEARNING_EXAMPLES_DIR}/training/nemo/llm/${MODEL}/run_nemo_${MODEL}.sh" \
python3 $envsubst_py -i ${TEMPLATE} -o pytorchjob.yaml

kubectl apply -f pytorchjob.yaml
