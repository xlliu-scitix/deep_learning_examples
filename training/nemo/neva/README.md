### Example scripts for Multimodal NeVa (LLaVA) pretraining

[NeVa(LLaVA)](https://docs.nvidia.com/nemo-framework/user-guide/latest/multimodalmodels/multimodallanguagemodel/neva/index.html),Originating from LLaVA (Large Language and Vision Assistant), NeVA is a groundbreaking addition to the NeMo Multimodal ecosystem. This model seamlessly integrates large language-centric models (like NVGPT or Llama2) with a vision encoder, and is trained with machine-generated multimodal language-image instruction-following data. This Example supports several LLM based NeVa. For Example, for llama2_70b_bf16, the folder provide two files:
- [neva_llama2_7b_chat_bf16_hydra.yaml](https://github.com/sallylxl/deep_learning_examples/blob/master/training/neva/neva_llama2_7b_chat_bf16_hydra.yaml)
  : recommended config for NeVa based on llama2_7b_chat with a vision encoder on NVIDIA H100, in fp16 data type, comming from [NeMo-Framework-Launcher](https://github.com/NVIDIA/NeMo-Framework-Launcher)
- [run_nemo_neva_llama2_7b_chat_bf16.sh](https://github.com/sallylxl/deep_learning_examples/blob/master/training/neva/run_nemo_neva_llama2_7b_chat_bf16.sh)
  : scripts run a recommended config for NeVa based on llama2_7b_chat with a vision encoder

#### Setup

1. To run these scripts, you must use nemo container (registry-ap-southeast.scitix.ai/hpc/nemo:24.07)

2. Dataset: use the LAION/CC/SBU BLIP-Caption Concept-balanced 558K dataset. You can get the dataset from Siflow preset datasets, or download the image data from [HuggingFace](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain). Update the config yaml if you need to add additional options.

#### Launch Pretraining

##### Launch Pretraining on Siflow

1. Update the following bash variables in the example running scripts, Optionally. For example, for ``neva_llama2_7b_chat_bf16`` pretrain, edit the running scripts [run_nemo_neva_llama2_7b_chat_bf16.sh](https://github.com/sallylxl/deep_learning_examples/blob/master/training/neva/run_nemo_neva_llama2_7b_chat_bf16.sh):

   - ``DEEP_LEARNING_EXAMPLES_DIR`` : the directory of where this repository is located, by default this is ``/data/deep_learning_examples``
   - ``BASE_RESULT_DIR`` : the directory of where the result of experiment is located, by default this is ``$DEEP_LEARNING_EXAMPLES_DIR/results``
   - ``PRETRAINED_LLM_PATH``: path to the LLM model checkpoints and tokenizer of NeMo’s format, by default this is the Siflow preset model path - ``/models/hf-to-nemo/llama2``
   - ``PRETRAINED_VISION_ENCODER_PATH``: path or name of the pretrained vison model, by default this is the Siflow preset model path - ``/models/openai/clip-vit-large-patch14-336``
   - ``DATASET_DIR`` - path to the pretrain dataset, by default this is the Siflow preset model path - ``/datasets/preset/liuhaotian/LLaVA-Pretrain``
   - ``WORLD_SIZE``: For Siflow Pytorchjob, the WORLD_SIZE is the number of GPU Nodes, will be set automatically.

2. Submit a Siflow Pytorchjob Task with the following command：
```
cd ${DEEP_LEARNING_EXAMPLES_DIR}/training/nemo/neva && ./run_nemo_neva_llama2_13b_chat_bf16.sh
```


##### Launch Pretraining on K8S Cluster

- [launcher_scripts/k8s/training/neva](https://github.com/sallylxl/deep_learning_examples/tree/master/launcher_scripts/k8s/training/neva)
  : Scripts to launch pytorchjob in a k8s cluster to run NeVa pretraining

1. Update the following bash variables in the example launch scripts. For example, for ``neva_llama2_7b_chat_bf16`` pretrain on a k8s, edit the lanuch scripts [launch_nemo_neva_llama2_13b_bf16.sh](https://github.com/sallylxl/deep_learning_examples/blob/master/launcher_scripts/k8s/training/nemo/neva/launch_nemo_neva_llama2_13b_bf16.sh):

   - ``DEEP_LEARNING_EXAMPLES_DIR`` : the directory of where this repository is located, by default this is ``/data/deep_learning_examples``
   - ``BASE_RESULT_DIR`` : the directory of where the result of experiment is located, by default this is ``$DEEP_LEARNING_EXAMPLES_DIR/results``
   - ``PRETRAINED_LLM_PATH``: path to the LLM model checkpoints and tokenizer of NeMo’s format, by default this is the Siflow preset model path - ``/data/uat/LLaVA/neva/checkpoints/llama-2``. You can also prepare the pretrained model checkpoints and tokenizer according to the guide from [NEMO_FRAMEWORK](https://docs.nvidia.com/nemo-framework/user-guide/latest/multimodalmodels/multimodallanguagemodel/neva/dataprep.html)
   - ``PRETRAINED_VISION_ENCODER_PATH``: path or name of the pretrained vison model, by default this is the Siflow preset model path ``/models/openai/clip-vit-large-patch14-336``. It will be downloaded automatically during pretraining. You can also download the model from [huggingface](https://huggingface.co/openai/clip-vit-large-patch14-336/tree/main) in advance，and specify this variable by model path.
   - ``DATASET_DIR`` - path to the pretrain dataset, by default this is the Siflow preset model path : ``/datasets/preset/liuhaotian/LLaVA-Pretrain``. You can also download the dataset from [huggingface](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/tree/main) 

2. Lanuch a Pytorchjob using the following command：
```
cd ${DEEP_LEARNING_EXAMPLES_DIR}/launcher_scripts/k8s/training/neva && ./launch_nemo_neva_llama2_13b_bf16.sh
```

### Pretraining Benchmark performance numbers (TBD)
