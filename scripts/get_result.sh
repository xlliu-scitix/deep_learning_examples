#!/bin/bash

#set -x

POD=`kubectl get pod |grep Runn |grep n80 |grep worker-8 |awk '{print $1}'`
kubectl logs $POD
kubectl logs $POD |grep "TFLOP/s/GPU" |grep -v "1/" |awk -F "TFLOP/s/GPU):" '{print $2}' |awk '{sum+=$1} END {if (NR > 0) printf "\n\navarage TFLOP/s/GPU=%.2f, %d records\n\n", sum/NR, NR}'


#TIME_AUTO=`kubectl logs ${POD} |grep RUN_ID= |awk -F = '{print $2}'`
TIME=${1:-$TIME_AUTO}

#cat /data/scitix/deep_learning_examples/results/meg_lm_llama2_70b_bf16/tp4_pp4_n80_gbs1280_mbs1_${TIME}/log-meg_lm_llama2_70b_bf16.log  |grep "TFLOP/s/GPU" |awk -F "TFLOP/s/GPU):" '{print $2}' |awk '{sum+=$1} END {if (NR>0) printf "\n\navarage TFLOP/s/GPU=%.2f, %d records\n\n", sum/NR, NR}'

