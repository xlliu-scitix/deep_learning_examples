### Example scripts for LLM pretraining

There are two files in each LLM model pretraining directory. For example, for Llama2 pretraining the corresponding folder provide the following sample scripts.
- [llama2_70b_bf16_hydra.yaml](https://github.com/sallylxl/deep_learning_examples/blob/master/training/nemo/llm/h100/llama2_70b_bf16/llama2_70b_bf16_hydra.yaml)
  : recommended config for llama2_70b on NVIDIA H100, in fp16 data type, comming from [NeMo-Framework-Launcher](https://github.com/NVIDIA/NeMo-Framework-Launcher)
- [run_nemo_lama2_70b_bf16.sh](https://github.com/sallylxl/deep_learning_examples/blob/master/training/nemo/llm/h100/llama2_70b_bf16/run_nemo_llama2_70b_bf16.sh)
  : scripts run a recommended config for llama2_70b

#### Setup

1. To run these scripts, you must use scitix nemo container ((registry-ap-southeast.scitix.ai/hpc/nemo:24.07))

2. For performance testing, the model config is configured with synthetic data. Update the config yaml if you need to add additional options.

#### Launch Pretraining

##### Launch Pretraining on Siflow

1. Update the following bash variables in the example running scripts, Optionally. For example, for ``gpt3_175b_2k_bf16`` pretrain, edit the running scripts [run_nemo_gpt3_175b_2k_bf16.sh](https://github.com/sallylxl/deep_learning_examples/blob/master/training/llm/h100/gpt3_175b_2k_bf16/run_nemo_gpt3_175b_2k_bf16.sh):

   - ``DEEP_LEARNING_EXAMPLES_DIR`` : the directory of where this repository is located, by default this is ``/data/deep_learning_examples``
   - ``BASE_RESULT_DIR`` : the directory of where the result of experiment is located, by default this is ``DEEP_LEARNING_EXAMPLES_DIR/results``
   - ``WORLD_SIZE``: For Siflow Pytorchjob, the WORLD_SIZE is the number of GPU Nodes, will be set automatically

2. Submit a Siflow Pytorchjob Task with the following command：
```
cd ${DEEP_LEARNING_EXAMPLES_DIR}/training/nemo/llm/h100/gpt3_175b_2k_bf16 && ./run_nemo_gpt3_175b_2k_bf16.sh
```


#### Launch Pretraining on K8S Cluster

- [launcher_scripts/k8s/training/llm](https://github.com/sallylxl/deep_learning_examples/tree/master/launcher_scripts/k8s/training/llm)
  : Scripts to launch pytorchjob in a k8s cluster to run LLM pretraining

1. Update the following bash variables in the example launch scripts. For example, for ``gpt3_175b_2k_bf16`` pretrain on a k8s, edit the lanuch scripts [launch_nemo_gpt3_175b_2k_bf16.sh](https://github.com/sallylxl/deep_learning_examples/blob/master/launcher_scripts/k8s/training/llm/lanuch_nemo_gpt3_175b_2k_bf16.sh):

   - ``DEEP_LEARNING_EXAMPLES_DIR`` : the directory of where this repository is located, by default this is ``/data/deep_learning_examples``
   - ``BASE_RESULT_DIR`` : the directory of where the result of experiment is located, by default this is ``DEEP_LEARNING_EXAMPLES_DIR/results``
   - ``WORLD_SIZE``: the number of GPU Nodes used to pretraining

2. Lanuch a Pytorchjob using the following command：
```
cd ${DEEP_LEARNING_EXAMPLES_DIR}/launcher_scripts/k8s/training/llm && ./launch_nemo_gpt3_175b_2k_bf16.sh
```

### Pretraining Benchmark performance numbers (TBD)
