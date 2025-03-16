from transformers import BitsAndBytesConfig
import torch
from .base import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from imports import LLAMA_MODEL_HF_DIR_PATH

class llama_text(BaseModel):
    def __init__(self, model_path='meta-llama/Meta-Llama-3-8B', **kwargs):
        self.device_map = kwargs["config"]["device_map"] if "config" in kwargs else "cuda"
        apply_quantization = kwargs["config"]["quantized"] if "config" in kwargs else False

        if kwargs["config"]["quant_type"] == "quant4":
            qunatization_config = bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        elif kwargs["config"]["quant_type"] == "quant8":
            qunatization_config = bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_quant_type="nf8",
                bnb_8bit_compute_dtype=torch.float16
            )

        processor = AutoTokenizer.from_pretrained(model_path,cache_dir=LLAMA_MODEL_HF_DIR_PATH)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=self.device_map, quantization_config=qunatization_config,cache_dir=LLAMA_MODEL_HF_DIR_PATH) if apply_quantization else AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=self.device_map,cache_dir=LLAMA_MODEL_HF_DIR_PATH)

        self.model = model
        self.processor = processor

        if "config" in kwargs:
            del kwargs['config']

    def generate_inner(self, message, dataset=None,category=None):
        content = ''
        for x in message:
            if x['type'] == 'text':
                content += x['value']

        inputs = self.processor(content, return_tensors="pt").to(device=self.device_map)
        attention_mask = inputs["attention_mask"]

        outputs = self.model.generate(
        inputs['input_ids'], 
        attention_mask=attention_mask,
        pad_token_id=self.processor.eos_token_id
        )

        return self.processor.decode(outputs[0], skip_special_tokens=True)
