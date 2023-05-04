python merge_llama_with_chinese_lora.py \
    --base_model decapoda-research/llama-7b-hf \
    --lora_model ziqingyang/chinese-alpaca-lora-7b \
    --output_type huggingface \
    --output_dir merged_model