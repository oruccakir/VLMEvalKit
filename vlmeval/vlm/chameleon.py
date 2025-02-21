import os.path as osp
import warnings
from .base import BaseModel
from ..smp import *
from PIL import Image
import torch

class Chameleon(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='facebook/chameleon-7b', **kwargs):
        try:
            from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
        except Exception as e:
            logging.critical('Please install the latest transformers.')
            raise e
        
        
        self.device_map = kwargs["device_map"] if "device_map" in kwargs else "cuda"
        apply_quantization = kwargs["apply_quantization"] if "apply_quantization" in kwargs is not None else False

        processor = ChameleonProcessor.from_pretrained(model_path)
        model = ChameleonForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=self.device_map, quantization_config=kwargs ["quant_config"]) if apply_quantization else ChameleonForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=self.device_map)

        self.model = model
        self.processor = processor

        if "apply_quantization" in kwargs:
            del kwargs["apply_quantization"]
        if "device_map" in kwargs:
            del kwargs['device_map']
        if "quant_config" in kwargs:
            del kwargs['quant_config']

    def generate_inner(self, message, dataset=None):
        content, images = '', []
        for x in message:
            if x['type'] == 'text':
                content += x['value']
            elif x['type'] == 'image':
                content += '<image>\n'
                images.append(Image.open(x['value']))

        inputs = self.processor(
            text=[content],
            images=images,
            padding=True,
            return_tensors='pt'
        ).to(device=self.device_map, dtype=torch.bfloat16)
        generate_ids = self.model.generate(**inputs, max_new_tokens=2048)
        input_token_len = inputs.input_ids.shape[1]
        text = self.processor.batch_decode(
            generate_ids[:, input_token_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return text
