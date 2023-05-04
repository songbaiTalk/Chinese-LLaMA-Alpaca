import torch
from peft import PeftConfig, PeftModel
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
peft_model_id = 'ziqingyang/chinese-alpaca-lora-7b'
peft_config = PeftConfig.from_pretrained(peft_model_id)