#!/bin/bash

set -x

POD=`kubectl get pod |grep Runn |grep n80 |grep worker-8 |awk '{print $1}'`
kubectl logs $POD
kubectl logs $POD |grep "TFLOP/s/GPU" |awk -F "TFLOP/s/GPU):" '{print $2}' |awk '{sum+=$1} END {if (NR > 0) printf "\n\navarage TFLOP/s/GPU=%.2f, %d records\n\n", sum/NR, NR}'
