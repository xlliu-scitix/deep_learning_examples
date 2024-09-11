# Example scripts for deep learing

This repository contains example scripts for deep learning, including pretraining configurations for Large Language Models (LLMs) and Multimodal Models. 

## Scripts Overview

### LLM Pretraining

- **[training/nemo/llm/h100](https://github.com/sallylxl/deep_learning_examples/tree/master/training/nemo/llm/h100)**  
  Scripts for LLM pretraining (e.g., GPT-3, LLaMA2, LLaMA3, NeMoTron) with recommended config (from [NeMo-Framework-Launcher](https://github.com/NVIDIA/NeMo-Framework-Launcher)) on NVIDIA H100, in fp16 data type, running on NeMo Framework
  - [llama2_13b_bf16](https://github.com/sallylxl/deep_learning_examples/tree/master/training/nemo/llm/h100/llama2_13b_bf16)
  - [llama2_70b_bf16](https://github.com/sallylxl/deep_learning_examples/tree/master/training/nemo/llm/h100/llama2_70b_bf16)
  - [llama3_1_70b_bf16](https://github.com/sallylxl/deep_learning_examples/tree/master/training/nemo/llm/h100/llama3_1_70b_bf16)
  - [gpt3_175b_2k_bf16](https://github.com/sallylxl/deep_learning_examples/tree/master/training/nemo/llm/h100/gpt3_175b_2k_bf16)
  - [nemotron_340b_bf16](https://github.com/sallylxl/deep_learning_examples/tree/master/training/nemo/llm/h100/nemotron_340b_bf16)

### Multimodal Models

- [training/nemo/neva](https://github.com/sallylxl/deep_learning_examples/tree/master/training/nemo/neva)
  : Scripts to Multimodal Models - NeVa (LLaVA) pretraining with recommended config (from [NeMo-Framework-Launcher](https://github.com/NVIDIA/NeMo-Framework-Launcher)) on NVIDIA H100, in fp16 data type, running on NeMo Framework

## Running Deep Learning Examples

### Prerequisites
Before running the examples, ensure the following:
- **Container**: Use the ScitiX NeMo container (`registry-ap-southeast.scitix.ai/hpc/nemo:24.07`) or the NGC NeMo container (`nemo:24.07`). If using NGC, clone this repository into the container or a shared storage accessible by distributed worker containers.
- **Datasets**: Refer to the README.md under [deep_learning_examples/training](https://github.com/sallylxl/deep_learning_examples/tree/master/training) for dataset preparation.
  - For LLM based on Megatron, mock data can be used.
  - For ScitiX SiFlow or CKS, preset datasets are available.
- **Pretrained Models**: Prepare corresponding pretrained models for fine-tuning and multimodal pretraining. Preset models are available for ScitiX SiFlow or CKS.

### Using [SiFlow All-in-One AI Platform](https://scitix.ai/SiflowService/index.aspx)

Refer to the README.md under [deep_learning_examples/training](https://github.com/sallylxl/deep_learning_examples/tree/master/training) for detailed instructions.

### Using [PyTorchjob Operator](https://github.com/kubeflow/pytorch-operator)
Scripts for launching PyTorch jobs on a Kubernetes cluster are located in [launcher_scripts/k8s](https://github.com/sallylxl/deep_learning_examples/tree/master/launcher_scripts/k8s/training).

Refer to the README.md under [deep_learning_examples/training](https://github.com/sallylxl/deep_learning_examples/tree/master/training) for detailed instructions.

For example, to launch the LLaMA2-13B pretraining, use the following command:

```
cd ${DEEP_LEARNING_EXAMPLES_DIR}/launcher_scripts/k8s/training/llm
./launch_nemo_llama2_13b_bf16.sh
```

## Performance
### LLM Training Performance Results
### NeVa Training Performance Results
### NeVa Finetune Performance Results
