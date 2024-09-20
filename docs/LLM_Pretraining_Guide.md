

# Prerequisites
## [deep_learning_examples repo](https://github.com/sallylxl/deep_learning_examples) 
Throughout the rest of this document, referenced files may be found in [deep_learning_example](https://github.com/sallylxl/deep_learning_examples) repo.

## Key Folders

+  **[NeMo LLM Pretraining Scripts](https://github.com/sallylxl/deep_learning_examples/tree/master/training/nemo/llm)** : Contains example scripts for pretraining GPT3 Models using the [NeMo Framework](ttps://docs.nvidia.com/nemo-framework/user-guide/latest/).  These scripts are adapted from [NeMo-Framework-Launcher](https://github.com/NVIDIA/NeMo-Framework-Launcher/tree/main)
+  **[Megatron-LM LLM Pretraining Scripts](https://github.com/sallylxl/deep_learning_examples/tree/master/training/Megatron-LM/llm/gpt3)** : Contains example scripts for pretraining GPT3 Models that adapted from [Megatron-LM](https://github.com/NVIDIA/Megatron-LM).
+  **[Megatron-DeepSpeed LLM Pretraining Scripts](https://github.com/sallylxl/deep_learning_examples/tree/master/training/Megatron-DeepSpeed/llm/gpt3)** : Contains example scripts for pretraining GPT3 Models that adapted from[Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed)
+ **[Kubernetes Launcher Scripts](https://github.com/sallylxl/deep_learning_examples/tree/master/launcher_scripts/k8s/training)**  :
   Includes [Pytorchjob](https://github.com/kubeflow/pytorch-operator) YAML files for Kubernetes resources and launcher scripts. Key specs of Pytorchjob YAML include:

  + Within this resource definition, there are two important `specs`: the `master` and the `worker`. the `master` and `worker` containers all run the same script using the same arguments. Both containers take the resources of an entire node, which includes 8 GPUs.
  + The PyTorch Job will also set up all the environment variables that are needed by `torchrun` and `dist` to set up distributed training, excluding `NODE_RANK`, which should be set using `RANK`. `MASTER_ADDR` and `MASTER_PORT` will point at the pod defined by the master spec.


## Dataset Preparation

### Synthetic Data

For testing performance, use synthetic data for pretrainning based NeMo and Megatron-LM

### Download and Pre-process Training Dataset
To use read dataset, before executing the steps below, you can download and pre-process the training set using the following commands (see [here](https://github.com/bigscience-workshop/Megatron-DeepSpeed?tab=readme-ov-file#quick-pre-processing-to-start-training-with) for more details):

```plain
wget https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
xz -d oscar-1GB.jsonl.xz
python tools/preprocess_data.py \
    --input oscar-1GB.jsonl \
    --output-prefix my-gpt2 \
    --vocab-file gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file gpt2-merges.txt \
    --append-eod \
    --workers 8
```

## Docker Image
Use either:
- ScitiX NeMo container: `registry-ap-southeast.scitix.ai/hpc/nemo:24.07`
- the NGC NeMo container: `nemo:24.07`. If using NGC, clone this repository into the container or a shared storage accessible by distributed worker containers

# Llama Pretraining Guide

## NeMo Llama Pretraining Guide

### Llama Pretraining Scripts

#### Llama Pretraining Python Script

The Llama pretraining python script is based Megatron-Core and adapted from the NeMo library [megatron_Llama_pretraining.py](https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/megatron_gpt_pretraining.py). It is available at container path: 
`/opt/NeMo/examples/nlp/language_modeling/megatron_Llama_pretraining.py`

#### Llama Pretraining Model Configurations

Recommended configuration for NVIDIA H100 GPUs using bf16 data type is available at (for `Llama2_13b_bf16` model):
[`${DEEP_LEARNING_EXAMPLES_DIR}/training/llm/nemo/llm/llama2_13b_bf16/llama2_13b_bf16_hydra.yaml`](https://github.com/sallylxl/deep_learning_examples/blob/master/training/nemo/llm/llama2_13b_bf16/llama2_13b_bf16_hydra.yaml).

#### Llama Pretraining Shell Script

The Llama pretraining shell script runs the above python script with following training parameters default (for `Llama2_13b_bf16` model):

+ The number of gradient accumulation microsteps is 2048, with micro batch size of 1.
+ The tensor parallelism degree is 4.
+ The pipeline parallel degree is 1.

The running script is available at: [`${DEEP_LEARNING_EXAMPLES_DIR}/training/llm/nemo/llm/llama2_13b_bf16/run_nemo_Llama2_13b_bf16.sh`](https://github.com/sallylxl/deep_learning_examples/blob/master/training/nemo/llm/llama2_13b_bf16/run_nemo_llama2_13b_bf16.sh)


### Initiating a Training Job
#### One-Click Test
Start a Training Job by applying the predefined `llama2-13b-bf16` PyTorchjob YAML:

```bash
kubectl apply -f ${DEEP_LEARNING_EXAMPLES_DIR}/launcher_scripts/k8s/training/nemo/llm/pytorchjob-llama2-13b-bf16-n8-gbs128-ckpt0.yaml
```

#### Launch Job Using PyTorchjob Operator

##### K8S Job Lancher Shell Script
The Llama training Lancher shell script runs the above **Llama Pretraining Shell Script** with following training parameters default (for `Llama2_13b_bf16` model) by launching a pytorchjob:

+ The number of gradient accumulation microsteps is 2048, with micro batch size of 1.
+ The tensor parallelism degree is 4.
+ The pipeline parallel degree is 1.
+ The number of workers is 1, each worker takes the resources of an entire node, which includes 8 GPUs.

The launch script `launch_nemo_llama2_13b_bf16.sh` is available at: [${DEEP_LEARNING_EXAMPLES_DIR}/launcher_scripts/k8s/training/nemo/llm/launch_nemo_llama2_13b_bf16.sh](https://github.com/sallylxl/deep_learning_examples/blob/master/launcher_scripts/k8s/training/nemo/llm/launch_nemo_llama2_13b_bf16.sh).

The `PyTorchJob` Kubernetes resource is defined in: [${DEEP_LEARNING_EXAMPLES_DIR}/launcher_scripts/k8s/training/nemo/llm/pytorchjob-nemo.yaml.template](https://github.com/sallylxl/deep_learning_examples/blob/master/launcher_scripts/k8s/training/nemo/llm/pytorchjob-nemo.yaml.template).


##### Step-by-Step Guide

1. **Update Variables**: Edit the example running script, taking `Llama2_13b_bf16` pretraining for example. Modify the following variables in [lancher_nemo_llama2_13b_bf16.sh](https://github.com/sallylxl/deep_learning_examples/blob/master/launcher_scripts/k8s/training/nemo/llm/launch_nemo_llama2_13b_bf16.sh):
   + `DEEP_LEARNING_EXAMPLES_DIR`: Path to this repository (default: `/workspace/deep_learning_examples`).
   + `BASE_RESULT_DIR`: Directory for experiment results (default: `$DEEP_LEARNING_EXAMPLES_DIR/results`).
   + `GPU_NUMS`: Number of GPUs used for distributed training.

2. **Lanuch Command**：

    ```plain
    cd ${DEEP_LEARNING_EXAMPLES_DIR}/launcher_scripts/k8s/training/nemo/llm
    ./launch_nemo_llama2_13b_bf16.sh
    ```

3. **Scaling Up**:
   If you want to scale up the number of GPUs used for distributed training, all you would need to do is set the number of `GPU_NUMS`，for example：

    ```plain
    cd ${DEEP_LEARNING_EXAMPLES_DIR}/launcher_scripts/k8s/training/nemo/llm
    GPU_NUMS=16 ./launch_nemo_llama2_13b_bf16.sh
    ```
4. **Monitor Training**:

    You can monitor the job's progress by checking the logs of the master pod. The outputs like below some lines showing throughput and loss statistics every log step.

    ```plain
    Epoch 0: :   2%|▏         | 2/128 [00:47<49:46, reduced_train_loss=10.60, global_step=1.0, consumed_samples=256, train_step_timing in s=23.7]
    ```


# GPT Pretraining Guide

## NeMo GPT Pretraining Guide
### GPT Pretraining Scripts
#### GPT Pretraining Python Script
The GPT pretraining python script is based Megatron-Core and adapted from the NeMo library [megatron_gpt_pretraining.py](https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/megatron_gpt_pretraining.py). 

The script is available at container path `/opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py`

#### GPT Model Configurations

Recommended configuration for NVIDIA H100 GPUs using bf16 data type is available at (for `gpt3_5b_2k_bf16` model):
[`${DEEP_LEARNING_EXAMPLES_DIR}/training/llm/nemo/llm/gpt3_5b_2k_bf16/gpt3_5b_2k_bf16_hydra.yaml`](https://github.com/sallylxl/deep_learning_examples/blob/master/training/nemo/llm/gpt3_5b_2k_bf16/gpt3_5b_2k_bf16_hydra.yaml).


#### GPT Pretraining Shell Script
The GPT training Lancher shell script runs the above python script with following training  parameters default (for `gpt3_5b_2k_bf16` model):

+ The number of gradient accumulation microsteps is 2048, with micro batch size of 1.
+ The tensor parallelism degree is 4.
+ The pipeline parallel degree is 8.

The script is available at [`${DEEP_LEARNING_EXAMPLES_DIR}/training/llm/nemo/llm/gpt3_5b_2k_bf16/run_nemo_gpt3_5b_2k_bf16.sh`](https://github.com/sallylxl/deep_learning_examples/blob/master/training/nemo/llm/gpt3_5b_2k_bf16/run_nemo_gpt3_5b_2k_bf16.sh)

### Initiating a Training Job
#### One-Click Test

you can start a Training Job by kubectl apply the predefined `gpt5-5b-2k-bf16` pytorchjob yaml by:

```bash
kubectl apply -f ${DEEP_LEARNING_EXAMPLES_DIR}/launcher_scripts/k8s/training/nemo/llm/pytorchjob-gpt3-5b-bf16-n8-gbs128-ckpt0.yaml
```
#### Launch Job Using PyTorchjob Operator

##### K8S Job Lancher Shell Script

The GPT training Lancher shell script runs the above **GPT Pretraining Shell Script** with following training parameters default (for `gpt3_5b_2k_bf16` model)  by lanching pytorchjob::

+ The number of gradient accumulation microsteps is 2048, with micro batch size of 1.
+ The tensor parallelism degree is 4.
+ The pipeline parallel degree is 8.
+ The number of workers is 8, each worker takes the resources of an entire node, which includes 8 GPUs.

The launch script `launch_nemo_gpt3_5b_2k_bf16.sh` is available at [deep_learning_examples/launcher_scripts/k8s/training/nemo/llm/launch_nemo_gpt3_5b_2k_bf16.sh](https://github.com/sallylxl/deep_learning_examples/blob/master/launcher_scripts/k8s/training/nemo/llm/launch_nemo_gpt3_5b_2k_bf16.sh).
The `PyTorchJob` Kubernetes resource is defined in [launcher_scripts/k8s/training/nemo/llm/pytorchjob-nemo.yaml.template](https://github.com/sallylxl/deep_learning_examples/blob/master/launcher_scripts/k8s/training/nemo/llm/pytorchjob-nemo.yaml.template).


##### Step-by-Step Guide

1. **Update Variables**: 
   Edit the example running script, taking `gpt3_5b_2k_bf16` pretraining for example. Modify the following variables in [lancher_nemo_gpt3_5b_2k_bf16.sh](https://github.com/sallylxl/deep_learning_examples/blob/master/launcher_scripts/k8s/training/nemo/llm/launch_nemo_gpt3_5b_2k_bf16.sh):
   + `DEEP_LEARNING_EXAMPLES_DIR`: Path to this repository (default: `/workspace/deep_learning_examples`).
   + `BASE_RESULT_DIR`: Directory for experiment results (default: `$DEEP_LEARNING_EXAMPLES_DIR/results`).
   + `GPU_NUMS`: Number of GPUs used for distributed training.

2. **Lanuch Command**：

    ```plain
    cd ${DEEP_LEARNING_EXAMPLES_DIR}/launcher_scripts/k8s/training/nemo/llm
    ./launch_nemo_gpt3_5b_2k_bf16.sh
    ```

3. **Scaling Up**:
   If you want to scale up the number of GPUs used for distributed training, all you would need to do is set the number of `GPU_NUMS`，for example：

    ```plain
    cd ${DEEP_LEARNING_EXAMPLES_DIR}/launcher_scripts/k8s/training/nemo/llm
    GPU_NUMS=16 ./launch_nemo_gpt3_5b_2k_bf16.sh
    ```
4. **Monitor Training**:

    You can monitor the job's progress by `kubectl logs` checking the logs of **the last RANK pod**. The outputs like below some lines showing throughput and loss statistics every log step.

    ```plain
    Epoch 0: :  45%|████▍     | 57/128 [1:12:51<1:30:44, reduced_train_loss=6.820, global_step=56.00, consumed_samples=1.17e+5, train_step_timing in s=74.40]
    Epoch 0: :  45%|████▌     | 58/128 [1:14:05<1:29:25, reduced_train_loss=6.820, global_step=56.00, consumed_samples=1.19e+5, train_step_timing in s=74.40]
    Epoch 0: :  46%|████▌     | 59/128 [1:15:19<1:28:05, reduced_train_loss=6.820, global_step=58.00, consumed_samples=1.21e+5, train_step_timing in s=74.40]
    Epoch 0: :  47%|████▋     | 60/128 [1:16:34<1:26:46, reduced_train_loss=6.720, global_step=59.00, consumed_samples=1.23e+5, train_step_timing in s=74.40]
    ```


## Megatron-LM GPT Pretraining Guide
### GPT Pretraining Scripts
#### GPT Pretraining Python Script
The GPT pretraining python script is based Megatron-Core and adapted from the Megatron-LM library [pretrain_gpt.py](https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py). 

The script is available at container path `$DEEP_LEARNING_EXAMPLES_DIR/thirdparty/Megatron-LM/pretrain_gpt.py`

#### GPT Pretraining Shell Script
The GPT training Lancher shell script runs the above python script with following training parameters default (for `gpt3_5b_2k_bf16` model):

+ The number of gradient accumulation microsteps is 2048, with micro batch size of 1.
+ The tensor parallelism degree is 4.
+ The pipeline parallel degree is 8.

The script is available at [`$DEEP_LEARNING_EXAMPLES_DIR/training/llm/megatron/llm/meg_lm_gpt3_5k_2k_bf16/run_meg_lm_gpt3_5k_2k_bf16.sh`](https://github.com/sallylxl/deep_learning_examples/blob/master/training/Megatron-LM/llm/gpt3/run_meg_lm_gpt3_5b_2k_bf16.sh)


### Initiating a Training Job
#### One-Click Test
you can start a Training Job by kubectl apply the predefined `meg_lm_gpt3_5b_2k_bf16` pytorchjob yaml by:

```bash
kubectl apply -f ${DEEP_LEARNING_EXAMPLES_DIR}/launcher_scripts/k8s/training/megatrion-lm/llm/pytorchjob-gpt3-5b-bf16-n8-gbs128-ckpt0.yaml
```
#### Launch Job Using PyTorchjob Operator
##### K8S Job Lancher Shell Script

The GPT training Lancher shell script runs the above Pretraining Shell Script with following training parameters default (for `meg_lm_gpt3_5b_2k_bf16` model)  by lanching pytorchjob::

+ The number of gradient accumulation microsteps is 2048, with micro batch size of 1.
+ The tensor parallelism degree is 4.
+ The pipeline parallel degree is 8.
+ The number of workers is 8, each worker takes the resources of an entire node, which includes 8 GPUs.

The launch script `launch_nemo_llama2_13b_bf16.sh` is available at [deep_learning_examples/launcher_scripts/k8s/training/nemo/llm/launch_nemo_llama2_13b_bf16.sh](https://github.com/sallylxl/deep_learning_examples/blob/master/launcher_scripts/k8s/training/nemo/llm/launch_nemo_llama2_13b_bf16.sh).
The `PyTorchJob` Kubernetes resource is defined in [deep_learning_examples/launcher_scripts/k8s/megatrion-lm/llm/pytorchjob.yaml.template](https://github.com/sallylxl/deep_learning_examples/blob/master/launcher_scripts/k8s/training/nemo/llm/pytorchjob-nemo.yaml.template).


##### Step-by-Step Guide

1. **Update Variables**: 
   Edit the example running script, taking `meg_lm_gpt3_5b_2k_bf16` pretraining for example. Modify the following variables in [launch_meg_lm_gpt3_5b_2k_bf16.sh](https://github.com/sallylxl/deep_learning_examples/blob/master/launcher_scripts/k8s/training/megatron-lm/llm/launch_meg_lm_gpt3_5b_2k_bf16.sh):
   + `DEEP_LEARNING_EXAMPLES_DIR`: Path to this repository (default: `/workspace/deep_learning_examples`).
   + `MOCK_DATA`: wheather use synthetic data to pretrain.  If use true data, update the following variable:
       - `DATA_DIR`: update your `DATA_DIR` to point to the pre-processed data (default: `/datasets/preset/bigscience/oscar-en`)
   + `BASE_RESULT_DIR`: Directory for experiment results (default: `$DEEP_LEARNING_EXAMPLES_DIR/results`).
   + `GPU_NUMS`: Number of GPUs used for distributed training.

2. **Lanuch Command**：

    ```plain
    cd ${DEEP_LEARNING_EXAMPLES_DIR}/launcher_scripts/k8s/training/megatron-lm/llm
    ./launch_meg_lm_gpt3_5b_2k_bf16.sh
    ```

3. **Scaling Up**:
   If you want to scale up the number of GPUs used for distributed training, all you would need to do is set the number of `GPU_NUMS`，for example：

    ```plain
    cd ${DEEP_LEARNING_EXAMPLES_DIR}/launcher_scripts/k8s/training/nemo/llm
    GPU_NUMS=256 ./launch_meg_lm_gpt3_5b_2k_bf16.sh
    ```

4. **Monitor Training**:

    You can monitor the job's progress by checking the logs of the master pod. The outputs like below some lines showing throughput and loss statistics every log step.


    ```plain
    `iteration     4873/   10000 | consumed samples:       311872 | elapsed time per iteration (ms): 8718.9 | learning rate: 1.500E-04 | global batch size:    64 | lm loss: 3.296875E+00 | grad norm: 0.430 | throughput: 7.340`
    ```

