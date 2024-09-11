# Example scripts for Multimodal NeVa (LLaVA) pretraining

This provides example scripts for pretraining [NeVa](https://docs.nvidia.com/nemo-framework/user-guide/latest/multimodalmodels/multimodallanguagemodel/neva/index.html) (NeMo Vision and Language Assistant) models, which integrate large language models (LLMs) with vision encoders. NeVa is a cutting-edge addition to the NeMo Multimodal ecosystem and is trained on multimodal language-image instruction-following data.

NeVa builds upon LLaVA (Large Language and Vision Assistant) and supports several LLM-based NeVa configurations. For example, the `llama2_7b_bf16` configuration includes two key files:

- **[neva_llama2_7b_chat_bf16_hydra.yaml](https://github.com/sallylxl/deep_learning_examples/blob/master/training/nemo/neva/neva_llama2_7b_chat_bf16_hydra.yaml)**  
  Recommended configuration for NeVa based on `llama2_7b_chat` with a vision encoder on NVIDIA H100, using fp16 data type. This configuration is sourced from [NeMo-Framework-Launcher](https://github.com/NVIDIA/NeMo-Framework-Launcher).

- **[run_nemo_neva_llama2_7b_chat_bf16.sh](https://github.com/sallylxl/deep_learning_examples/blob/master/training/nemo/neva/run_nemo_neva_llama2_7b_chat_bf16.sh)**  
  Script to run the recommended configuration for NeVa based on `llama2_7b_chat` with a vision encoder.


## Prerequisites

1. **Container**: Use either the ScitiX NeMo container (`registry-ap-southeast.scitix.ai/hpc/nemo:24.07`) or the NGC NeMo container (`nemo:24.07`). If using NGC, clone this repository into the container or a shared storage accessible by distributed worker containers.
2. **Dataset**: Use the LAION/CC/SBU BLIP-Caption Concept-balanced 558K dataset. You can obtain this from Siflow preset datasets or download the image data from [HuggingFace](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain). Update the configuration YAML file if additional options are needed.


## Launch Pretraining

### Using [SiFlow All-in-One AI Platform](https://scitix.ai/SiflowService/index.aspx)

1. **Update Variables**: Edit the example script, taking `neva_llama2_7b_chat_bf16` pretraining for example. Modify the following variables in [run_nemo_neva_llama2_7b_chat_bf16.sh](https://github.com/sallylxl/deep_learning_examples/blob/master/training/nemo/neva/run_nemo_neva_llama2_7b_chat_bf16.sh):
   - `DEEP_LEARNING_EXAMPLES_DIR`: Path to the repository (default: `/workspace/deep_learning_examples`).
   - `BASE_RESULT_DIR`: Directory for experiment results (default: `$DEEP_LEARNING_EXAMPLES_DIR/results`).
   - `PRETRAINED_LLM_PATH`: Path to LLM model checkpoints and tokenizer (default: Siflow preset model path - `/models/preset/scitix/hf-to-nemo/Llama-2-7b-chat`).
   - `PRETRAINED_VISION_ENCODER_PATH`: Path or name of the pretrained vision model (default: Siflow preset model path - `/models/presetopenai/clip-vit-large-patch14-336`).
   - `DATASET_DIR`: Path to the pretraining dataset (default: Siflow preset model path - `/datasets/preset/liuhaotian/LLaVA-Pretrain-LCS-558K`).
   - `WORLD_SIZE`: Number of GPU nodes for Siflow PyTorch job (automatically set).

2. **Submit a Siflow PyTorch Job**: with the following command:
   ```bash
   cd ${DEEP_LEARNING_EXAMPLES_DIR}/training/nemo/neva
   ./run_nemo_neva_llama2_13b_chat_bf16.sh
   ```


### Using [PyTorchjob Operator](https://github.com/kubeflow/pytorch-operator)

- [launcher_scripts/k8s/training/neva](https://github.com/sallylxl/deep_learning_examples/tree/master/launcher_scripts/k8s/training/nemo/neva)
  : Scripts to launch pytorchjob on a k8s cluster for NeVa pretraining

1. **Update Variables**: Edit the example script, taking `neva_llama2_7b_chat_bf16` pretraining for example. Modify the following variables in [launch_nemo_neva_llama2_7b_bf16.sh](https://github.com/sallylxl/deep_learning_examples/blob/master/launcher_scripts/k8s/training/nemo/neva/launch_nemo_neva_llama2_7b_bf16.sh):
   - `DEEP_LEARNING_EXAMPLES_DIR`: Path to the repository (default: `/workspace/deep_learning_examples`).
   - `BASE_RESULT_DIR`: Directory for experiment results (default: `$DEEP_LEARNING_EXAMPLES_DIR/results`).
   - `PRETRAINED_LLM_PATH`: Path to LLM model checkpoints and tokenizer (default: Siflow preset model path - `/models/preset/scitix/hf-to-nemo/Llama-2-7b-chat`). Refer to the [Nemo Framework documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/multimodalmodels/multimodallanguagemodel/neva/dataprep.html)for preparing model checkpoints. 
   - `PRETRAINED_VISION_ENCODER_PATH`: Path or name of the pretrained vision model (default: Siflow preset model path - `/models/preset/openai/clip-vit-large-patch14-336`). It will be downloaded automatically or you can pre-download from [huggingface](https://huggingface.co/openai/clip-vit-large-patch14-336/tree/main) 
   - `DATASET_DIR`: Path to the pretraining dataset (default: Siflow preset model path - `/datasets/preset/liuhaotian/LLaVA-Pretrain-LCS-558K`). You can also download it from [huggingface](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/tree/main) 
   - `WORLD_SIZE`: Number of GPU nodes for Siflow PyTorch job (automatically set).


2. Lanuch a Pytorchjob using the following command：
  ```
  cd ${DEEP_LEARNING_EXAMPLES_DIR}/launcher_scripts/k8s/training/nemo/neva
  ./launch_nemo_neva_llama2_7b_bf16.sh
  ```
