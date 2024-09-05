### Example scripts for deep learing

These scripts run a recommended config for LLM, Multimodal Models pretraining, including:

- [training/nemo/llm/h100](https://github.com/sallylxl/deep_learning_examples/tree/master/training/nemo/llm/h100)
  : Scripts to run LLM (gpt3, llama2, llama3, nemotron) pretraining with recommended config on NVIDIA H100, in fp16 data type, running on NeMo Framework
  - [gpt3_175b_2k_bf16](https://github.com/sallylxl/deep_learning_examples/tree/master/training/nemo/llm/h100/gpt3_175b_2k_bf16)
  - [llama2_70b_bf16](https://github.com/sallylxl/deep_learning_examples/tree/master/training/nemo/llm/h100/llama2_70b_bf16)
  - [llama3_1_70b_bf16](https://github.com/sallylxl/deep_learning_examples/tree/master/training/nemo/llm/h100/llama3_1_70b_bf16)
  - [nemotron_340b_bf16](https://github.com/sallylxl/deep_learning_examples/tree/master/training/nemo/llm/h100/nemotron_340b_bf16)
- [training/nemo/neva](https://github.com/sallylxl/deep_learning_examples/tree/master/training/nemo/neva)
  : Scripts to run Multimodal Models - NeVa (LLaVA) pretraining with recommended config on NVIDIA H100, in fp16 data type, running on NeMo Framework

- [launcher_scripts/k8s/training](https://github.com/sallylxl/deep_learning_examples/tree/master/launcher_scripts/k8s/training)
  : Scripts to launch pytorchjob in a k8s cluster to run the deep learing examples