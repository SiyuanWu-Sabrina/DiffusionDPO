# Prompt Reconstruction

> 吴斯媛
>
> Last Modified: 2024/3/15



### 文档结构

```bash
.
├── data
│   ├── generated_dalle3_prompts
|   │   ├── 0
│   │   │   ├── dalle3_revised_prompts_reconstructed_filtered_basic_test.csv
│   │   │   ├── dalle3_revised_prompts_reconstructed_filtered_basic_train.csv
│   │   │   └── dalle3_revised_prompts_reconstructed_filtered_basic_val.csv
│   │   ├── 1
│   │   │   ├── dalle3_revised_prompts_reconstructed_filtered_basic_test.csv
│   │   │   ├── dalle3_revised_prompts_reconstructed_filtered_basic_train.csv
│   │   │   └── dalle3_revised_prompts_reconstructed_filtered_basic_val.csv
│   │   ...
│   ├── dalle3_prompts_orig.txt
│   ├── dalle3_revised_prompts_orig
│   │   ├── basic_test_revised_prompts.txt
│   │   ├── basic_train_revised_prompts.txt
│   │   └── basic_val_revised_prompts.txt
│   └── midjourney_prompts_orig.txt
├── output
│   ├── 0
│   │   ├── dalle3_reconstruction_error_log-basic_test.txt
│   │   ├── dalle3_reconstruction_error_log-basic_train.txt
│   │   ├── dalle3_reconstruction_error_log-basic_val.txt
│   │   ├── dalle3_reconstruction_process_log-basic_test.txt
│   │   ├── dalle3_reconstruction_process_log-basic_train.txt
│   │   └── dalle3_reconstruction_process_log-basic_val.txt
│   ├── 1
│   │   ├── ...
│   │   ├── ...
│   │   ├── ...
│   │   ├── ...
│   │   ├── ...
│   │   └── ...
│   └── ...
├── config.py
├── data_process.py
├── reconstruct.py
├── script
│   └── reconstruct.sh
└── requirements.txt
```

#### 代码

- `config.py` 定义了原始数据集的路径、将原始数据集中的 prompt 统一提取之后保存的文件路径、ChatGLM 的参数路径、以及用于修改原始 prompt 的提示词。
- `preprocess.py` 用于数据预处理，将原始数据集中的 prompt 统一提取并保存到目标文件中，避免重复读取原始数据集。
- `reconstruct.py` 用于重构 prompt，包含主函数 `reconstruction_with_chatglm3_6b`，可以自定义重构参数。
- `script/reconstruct.sh` 入口脚本。

#### 数据

- `data`：保存了统一提取的原始 prompt 和 ChatGLM 依照某些提示词重写的 prompt：
  - `dalle3_prompts_orig.txt`、`dalle3_revised_prompts_orig` 和 `midjourney_prompts_orig.txt` 中是原始 prompt；
  - `{id}/{filename}.csv` 中存放的是 ChatGLM 依照某些提示词重写的 prompt。其中 `id` 对应的输入给 ChatGLM 的提示词即 `config.py` 中定义的 `PROMPT_MAPPING` 的 `key`。如需修改、新增新的 prompt，需要在已有的列表后增加。
- `output`：保存了脚本运行过程的输出，可以用于实时监控生成效果，目录结构类似 `data`。

#### 依赖

- `requirements.txt`



### 用法

```bash
cd /share/imagereward_work/prompt_reconstruction/
pip install -r requirements.txt
bash script/reconstruct.sh $PROMPT_ID$
# you should change $PROMPT_ID$ to an integer, which would serve as the key for 'PROMPT_MAPPING' in config.py
```


### 参考

- ChatGLM 参数路径：`/share/official_pretrains/hf_home/chatglm3-6b`