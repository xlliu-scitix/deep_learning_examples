#!/bin/bash

set -x

#for host in `cat hostfile`;do 
#	#ssh root@$host "apt-get install cpufrequtils -y && cpufreq-set -r -g performance"
#	ssh root@$host "echo performance |tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
#done

for host in `cat hostfile`;do 
	ssh root@$host "cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
done
