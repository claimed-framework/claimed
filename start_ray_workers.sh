#!/bin/bash
#from https://github.ibm.com/mssv/llm-eval-with-raytune/blob/main/ccc/ray_start_workers.sh
obj_store_mem="10000000000" #10G

while getopts ":a:e:" option;do
    case "${option}" in
    a) a=${OPTARG}
        head_address=$a
    ;;
    e) e=${OPTARG}
        conda_env=$e
    ;;
    *) echo "Did not supply the correct arguments"
    ;;
    esac
    done

if [ -z "$head_address" ]
then
    echo "must provide head node address via -a flag, e.g. -a 9.47.193.68:20022, exiting..."
    exit 1
fi

source ~/.bashrc

if [ -z "$conda_env" ]
then
    echo "Not activating any conda env"        
fi

conda activate $conda_env
if [ $? -ne 0 ]; then
    echo "failed to activate conda environment, exiting..."
    exit 1
fi

trap cleanup EXIT
function cleanup()
{
    echo "stopping ray workers on $LSB_MCPU_HOSTS"
    blaunch -z "$LSB_MCPU_HOSTS" source ~/.bashrc && conda activate $conda_env && ray stop
}

hosts=()
for host in `cat $LSB_DJOB_HOSTFILE | uniq`
do
        echo "Adding host: " $host
        hosts+=($host)
done

echo "The host list is: " "${hosts[@]}"

echo "Head node address is $head_address" 
export head_address

port=20022
export port

echo "Num cpus per host is:" $LSB_MCPU_HOSTS
IFS=' ' read -r -a array <<< "$LSB_MCPU_HOSTS"
declare -A associative
i=0
len=${#array[@]}
while [ $i -lt $len ]
do
    key=${array[$i]}
    value=${array[$i+1]}
    associative[$key]+=$value
    i=$((i=i+2))
done
# parse the number of GPUs per node, SLURM_JOB_GPUS looks something like "2,3,4" for 3 GPUs
if [ -z "$CUDA_VISIBLE_DEVICES" ]
then
    num_gpu=0
else
    num_gpu=`echo -n $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}'`
fi

workers=("${hosts[@]:0}")

echo "adding the workers to head node: " "${workers[*]}"
#run ray on worker nodes and connect to head
for host in "${workers[@]}"
do
    echo "starting worker on: " $host "and using master node: " $head_address
    num_cpu=${associative[$host]}
    command_for_worker="blaunch -z $host ray  start --address $head_address --num-cpus $num_cpu --num-gpus $num_gpu --object-store-memory $obj_store_mem"
    echo "launching worker on $host: $command_for_worker"
    $command_for_worker &
    sleep 5
done

# block
sleep infinity
