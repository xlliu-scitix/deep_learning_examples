# Example scripts for LLM pretraining

This provides example scripts for pretraining Large Language Models (LLMs). Taking `llama2_13b_bf16` for example, each LLM model pretraining directory contains two key files:


- **[llama2_13b_bf16_hydra.yaml](https://github.com/sallylxl/deep_learning_examples/blob/master/training/nemo/llm/llama2_13b_bf16/llama2_13b_bf16_hydra.yaml)**  
  Recommended configuration for `llama2_13b_bf16` on NVIDIA H100 GPUs, using fp16 data type. This config is sourced from [NeMo-Framework-Launcher](https://github.com/NVIDIA/NeMo-Framework-Launcher).

- **[run_nemo_llama2_13b_bf16.sh](https://github.com/sallylxl/deep_learning_examples/blob/master/training/nemo/llm/llama2_13b_bf16/run_nemo_llama2_13b_bf16.sh)**  
  Script to run the recommended configuration for `llama2_13b_bf16`.


## Prerequisites

1. **Container**: Use either the ScitiX NeMo container (`registry-ap-southeast.scitix.ai/hpc/nemo:24.07`) or the NGC NeMo container (`nemo:24.07`). If using NGC, clone this repository into the container or a shared storage accessible by distributed worker containers.

2. **Datasets**: For performance testing, the model is configured with synthetic data. Update the config YAML if additional options are needed.

## Launch Pretraining

### Using [SiFlow All-in-One AI Platform](https://scitix.ai/SiflowService/index.aspx)


1. **Update Variables**: Edit the example running script, taking `llama2_13b_bf16` pretraining for example. Modify the following variables in [run_nemo_gpt3_175b_2k_bf16.sh](https://github.com/sallylxl/deep_learning_examples/blob/master/training/nemo/llm/llama2_13b_bf16/run_nemo_llama2_13b_bf16.sh):
   - `DEEP_LEARNING_EXAMPLES_DIR`: Path to this repository (default: `/workspace/deep_learning_examples`).
   - `BASE_RESULT_DIR`: Directory for experiment results (default: `$DEEP_LEARNING_EXAMPLES_DIR/results`).
   - `WORLD_SIZE`: Number of GPU nodes for Siflow PyTorch job (automatically set).

2. **Submit a Siflow PyTorch Job**: Use the following command to run the pretraining:
   ```bash
   cd ${DEEP_LEARNING_EXAMPLES_DIR}/training/nemo/llm/llama2_13b_bf16
   ./run_nemo_llama2_13b_bf16.sh
   ```

### Using [PyTorchjob Operator](https://github.com/kubeflow/pytorch-operator)

- [launcher_scripts/k8s/training/llm](https://github.com/sallylxl/deep_learning_examples/tree/master/launcher_scripts/k8s/training/nemo/llm)
  : Scripts to launch pytorchjob in a k8s cluster to run LLM pretraining

1. **Update Variables**: Edit the example running script, taking `llama2_13b_bf16` pretraining for example. Modify the following variables in [launch_nemo_llama2_13b_bf16.sh](https://github.com/sallylxl/deep_learning_examples/blob/master/launcher_scripts/k8s/training/nemo/llm/launch_nemo_llama2_13b_bf16.sh):
   - `DEEP_LEARNING_EXAMPLES_DIR`: Path to the repository (default: `/workspace/deep_learning_examples`).
   - `BASE_RESULT_DIR`: Directory for experiment results (default: `$DEEP_LEARNING_EXAMPLES_DIR/results`).
   - `WORLD_SIZE`: Number of GPU nodes for Siflow PyTorch job (automatically set).

2. Lanuch a Pytorchjob using the following command：
  ```
  cd ${DEEP_LEARNING_EXAMPLES_DIR}/launcher_scripts/k8s/training/nemo/llm
  ./launch_nemo_llama2_13b_bf16.sh
  ```

