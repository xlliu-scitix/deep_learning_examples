#!/bin/bash

set -x

i=0
for host in `cat hostfile`;do 
	ssh root@$host "docker run -d --name=llama2-70b-test-onetime --gpus all --device=/dev/infiniband --network=host --cap-add=IPC_LOCK -v /data/scitix/deep_learning_examples:/workspace/deep_learning_examples --rm -v /dev/shm:/dev/shm registry-ap-southeast.scitix.ai/hpc/nemo:24.07  bash -c \"export NODE_RANK=$i && export WORLD_SIZE=10 && export MASTER_ADDR=10.208.54.27 && timeout 120 bash /workspace/deep_learning_examples/run_cmd.sh || sleep 10 && timeout 600 bash /workspace/deep_learning_examples/run_cmd.sh\""
	((i++))
done
