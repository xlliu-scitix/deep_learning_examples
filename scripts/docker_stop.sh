#!/bin/bash

set -x

for host in `cat hostfile`;do 
	ssh root@$host docker stop llama2-70b-test-onetime &
done
