#!/bin/bash

set -x

#for host in `cat hostfile`;do 
#	ssh root@$host "nvidia-smi -pm 1"
#done

for host in `cat hostfile`;do 
	ssh root@$host "nvidia-smi --query-gpu=persistence_mode --format=csv,noheader"
done
for host in `cat hostfile`;do 
	ssh root@$host "lsmod |grep peer"
done
for host in `cat hostfile`;do 
	ssh root@$host "lspci -vvv |grep ACSCtl"
done
