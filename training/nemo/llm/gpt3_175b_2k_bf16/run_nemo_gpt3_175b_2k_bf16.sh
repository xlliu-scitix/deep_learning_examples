#!/bin/bash

set -ex

# setup env
export NCCL_IB_TIMEOUT=22
export TRANSFORMERS_OFFLINE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NVTE_DP_AMAX_REDUCE_INTERVAL=0
export NVTE_ASYNC_AMAX_REDUCTION=1
export NVTE_FUSED_ATTN=0
export NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_NET_GDR_LEVEL=3
if [ -n "$RANK" ];then
  export NODE_RANK=${RANK}
  unset RANK
fi

# set params
#UB_TP_COMM_OVERLAP=${UB_TP_COMM_OVERLAP:-False}
#export TOKENIZERS_PARALLELISM=${UB_TP_COMM_OVERLAP}

# setup workspace dir and base result dir
MODEL="gpt3_175b_2k_bf16"
DEEP_LEARNING_EXAMPLES_DIR=${DEEP_LEARNING_EXAMPLES_DIR:-/workspace/deep_learning_examples}
BASE_RESULTS_DIR=${BASE_RESULTS_DIR:-${DEEP_LEARNING_EXAMPLES_DIR}/results}

# setup training parameters
WORLD_SIZE=${WORLD_SIZE:-16}
TP=${TP:-4}
PP=${PP:-8}
VPP=${VPP:-6}
CP=${CP:-1}
GBS=${GBS:-$((128*WORLD_SIZE))}
MBS=${MBS:-1}
MAX_STEPS=${MAX_STEPS:-128}

# setup experiment result dir
MODEL_DIR=${MODEL_DIR:-"${DEEP_LEARNING_EXAMPLES_DIR}/training/nemo/llm/${MODEL}"}
CURR_TIME=$(date +"%m%dT%H") # not %H%M as the start times of different workers may vary by several minutes
RUN_ID=${RUN_ID:-${CURR_TIME}}
RESULTS_DIR=${BASE_RESULTS_DIR}/${MODEL}/tp${TP}_pp${PP}_cp${CP}_n$((WORLD_SIZE * 8))_gbs${GBS}_mbs${MBS}_${RUN_ID}

# Check if the world_size is divisable by TP * PP * CP
global_world_size=$((WORLD_SIZE * 8))
divisor=$((TP * PP * CP))
if (( global_world_size % divisor != 0 )); then
	echo "global_world_size ${global_world_size} is not divisible by TP ${TP} * PP ${PP} * CP ${CP}"
	exit 1
fi

# Check if the GBS is divisable by MBS * DP
DP=$((global_world_size / divisor))
divisor=$((DP * MBS))
if (( GBS % divisor != 0 )); then
	echo "global batch size ${GBS} is not divisible by micro batch size (${MBS}) times data parallel size (${DP})"
	exit 1
fi

# generate the config file.
# If there is no NFS, generate the config file on each rank. Otherwise, generate the config file on rank 0.
SCRIPT_DIR=$(realpath $(dirname $0)) # Get the directory of the current script
envsubst_py=$(echo "$SCRIPT_DIR" |awk -F 'deep_learning_examples' '{print $1"/deep_learning_examples/launcher_scripts/envsubst.py"}')
NFS=${NFS:-True}
if [ $NODE_RANK -eq 0 ] || [ "x${NFS}" != "xTrue" ] ;then
        mkdir -p ${RESULTS_DIR}
        ENABLE_CKPT=${ENABLE_CKPT:-False} \
	UB_TP_COMM_OVERLAP=${UB_TP_COMM_OVERLAP} \
        WORLD_SIZE=$WORLD_SIZE GBS=$GBS MBS=$MBS PP=$PP VPP=$VPP TP=$TP CP=$CP MAX_STEPS=$MAX_STEPS RESULTS_DIR=${RESULTS_DIR} \
                python3 $envsubst_py -i ${MODEL_DIR}/${MODEL}_hydra.yaml -o ${RESULTS_DIR}/${MODEL}_hydra.yaml
else
        while [ ! -f "${RESULTS_DIR}/${MODEL}_hydra.yaml" ]; do
                echo "${RESULTS_DIR}/${MODEL}_hydra.yaml not exist, waiting..."
                sleep 5
        done
fi

# command 1
bash -c "
  cd /opt/NeMo;
  git rev-parse HEAD;
  export PYTHONPATH=/opt/NeMo:\${PYTHONPATH};
  (echo PYT$"NVIDIA_PYTORCH_VERSION" &&                 git --git-dir=/opt/NeMo/.git log -n 5 --format='NeMo;%h;%aD;%s' &&                 git --git-dir=/opt/megatron-lm/.git log -n 5 --format='megatron-lm;%h;%aD;%s') > ${RESULTS_DIR}/git_log.txt;
  CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NVTE_FWD_LAYERNORM_SM_MARGIN=\$(python3 /opt/NeMo-Framework-Launcher/launcher_scripts/nemo_launcher/collections/conditional_cfgs.py name=get_ln_sm_margin) \
  torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=8 --rdzv-backend=c10d --rdzv-endpoint=${MASTER_ADDR:-127.0.0.1} /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py  \
  --config-path=${RESULTS_DIR} \
  --config-name=${MODEL}_hydra.yaml \
  model.gc_interval=100 " 2>&1 | tee ${RESULTS_DIR}/log_${MODEL}_${RUN_ID}.out
