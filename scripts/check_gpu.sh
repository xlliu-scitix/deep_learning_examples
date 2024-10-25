#!/bin/bash

set -x

for host in `cat hostfile`;do 
	ssh root@$host "/data/scitix/tmp/ugmi/ugmi -dt gpu -tc mon"
done
