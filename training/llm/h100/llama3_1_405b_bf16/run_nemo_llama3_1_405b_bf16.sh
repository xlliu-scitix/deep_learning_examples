#!/bin/bash

set -x

# setup
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
MODEL="llama3_1_405b_bf16"
DEEP_LEARNING_EXAMPLES_DIR=${DEEP_LEARNING_EXAMPLES_DIR:-/data/deep_learning_examples}
BASE_RESULTS_DIR=${BASE_RESULTS_DIR:-${DEEP_LEARNING_EXAMPLES_DIR}/results}

# setup training parameters
WORLD_SIZE=${WORLD_SIZE:-144}
TP=${TP:-8}
PP=${PP:-9}
VPP=${VPP:-7}
CP=${CP:-2}
GBS=${GBS:-2016}
MBS=${MBS:-1}
SEQ_LEN={$SEQ_LEN:-8192}
MAX_STEPS=${MAX_STEPS:-128}

# setup experiment result dir
MODEL_DIR=${MODEL_DIR:-"${DEEP_LEARNING_EXAMPLES_DIR}/training/llm/h100/${MODEL}"}
CURR_TIME=$(date +"%m%dT%H%M")
RUN_ID=${RUN_ID:-${CURR_TIME}}
RESULTS_DIR=${BASE_RESULTS_DIR}/${MODEL}/tp${TP}_pp${PP}_cp${CP}_n$((WORLD_SIZE * 8))_gbs${GBS}_mbs${MBS}_${RUN_ID}

# Check if the world_size is divisable by TP * PP * CP
global_world_size=$((WORLD_SIZE * 8))
divisor=$((TP * PP * CP))
if (( global_world_size % divisor != 0)); then
	echo "global_world_size ${global_world_size} is not divisible by TP ${TP} * PP ${PP} * CP ${CP}"
	exit 1
fi

# Check if the GBS is divisable by MBS * DP * PP
dp=$((global_world_size / divisor))
divisor=$((dp * MBS * PP))
if (( GBS % divisor != 0)); then
	echo "global batch size ${GBS} is not divisible by micro batch size (${MBS}) times data parallel size (${DP})"
	exit 1
fi

# generate the config file.
# If there is no NFS, generate the config file on each rank. Otherwise, generate the config file on rank 0.
NFS=${NFS:-True}
if [ $NODE_RANK -eq 0 ] || [ "x${NFS}" == "x" ] ;then
        mkdir -p ${RESULTS_DIR}
        ENABLE_CKPT=${ENABLE_CKPT:-False} \
	UB_TP_COMM_OVERLAP=${UB_TP_COMM_OVERLAP} \
        WORLD_SIZE=$WORLD_SIZE GBS=$GBS MBS=$MBS PP=$PP VPP=$VPP TP=$TP CP=$CP SEQ_LEN=$SEQ_LEN MAX_STEPS=$MAX_STEPS RESULTS_DIR=${RESULTS_DIR} \
                /data/envsubst < ${MODEL_DIR}/${MODEL}_hydra.yaml  > ${RESULTS_DIR}/${MODEL}_hydra.yaml
else
        sleep 5
fi

# command 1
bash -c "
  cd /opt/NeMo;
  git rev-parse HEAD;
  export PYTHONPATH=/opt/NeMo:\${PYTHONPATH};
  (echo PYT$"NVIDIA_PYTORCH_VERSION" &&                 git --git-dir=/opt/NeMo/.git log -n 5 --format='NeMo;%h;%aD;%s' &&                 git --git-dir=/opt/megatron-lm/.git log -n 5 --format='megatron-lm;%h;%aD;%s') > ${RESULTS_DIR}/git_log.txt;
  CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NVTE_FWD_LAYERNORM_SM_MARGIN=\$(python3 /opt/NeMo-Framework-Launcher/launcher_scripts/nemo_launcher/collections/conditional_cfgs.py name=get_ln_sm_margin) NVTE_BWD_LAYERNORM_SM_MARGIN=\$(python3 /opt/NeMo-Framework-Launcher/launcher_scripts/nemo_launcher/collections/conditional_cfgs.py name=get_ln_sm_margin) NVTE_UB_SPLIT_AG=\$(python3 /opt/NeMo-Framework-Launcher/launcher_scripts/nemo_launcher/collections/conditional_cfgs.py name=get_ag_overlap fp8=False ) \
  torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=8 --rdzv-backend=c10d --rdzv-endpoint=${MASTER_ADDR:-127.0.0.1} /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py  \
  --config-path=${RESULTS_DIR} \
  --config-name=${MODEL}_hydra.yaml \
  model.gc_interval=100 " 2>&1 | tee ${RESULTS_DIR}/log_${MODEL}_${RUN_ID}.out
