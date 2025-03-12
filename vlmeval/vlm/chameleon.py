import os.path as osp
import warnings
from .base import BaseModel
from ..smp import *
from PIL import Image
import torch

from transformers import BitsAndBytesConfig
from imports import CHAMELEON_MODEL_EMBEDDINS_DIR_PATH

class Chameleon(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='facebook/chameleon-7b', **kwargs):
        try:
            from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
        except Exception as e:
            logging.critical('Please install the latest transformers.')
            raise e
        
        
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

        processor = ChameleonProcessor.from_pretrained(model_path)
        model = ChameleonForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=self.device_map, quantization_config=qunatization_config) if apply_quantization else ChameleonForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=self.device_map)

        self.model = model
        self.processor = processor

        if "apply_quantization" in kwargs:
            del kwargs["apply_quantization"]
        if "device_map" in kwargs:
            del kwargs['device_map']
        if "quant_config" in kwargs:
            del kwargs['quant_config']

        self.idx = 0

    def generate_inner(self, message, dataset=None):
        content, images = '', []
        for x in message:
            if x['type'] == 'text':
                content += x['value']
            elif x['type'] == 'image':
                content += '<image>\n'
                images.append(Image.open(x['value']))

        embedd_dir_path=f"{CHAMELEON_MODEL_EMBEDDINS_DIR_PATH}/{dataset}"
        if not os.path.exists(embedd_dir_path):
            os.makedirs(embedd_dir_path)
        
        embedding_file_path = f"{embedd_dir_path}/embedding_{self.idx}.bin"
        self.idx += 1
        
    
        inputs = self.processor(
            text=[content],
            images=images,
            padding=True,
            return_tensors='pt'
        ).to(device=self.device_map, dtype=torch.bfloat16)

        self.compute_and_save_embeddings(inputs,embedding_file_path)

        generate_ids = self.model.generate(**inputs, max_new_tokens=2048)
        input_token_len = inputs.input_ids.shape[1]
        text = self.processor.batch_decode(
            generate_ids[:, input_token_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return text
    
    def compute_and_save_embeddings(self, inputs, embedding_file_path):
        self.model.model.save_embedding_flag = True
        self.model.model.embedding_file_path = embedding_file_path
        try :
            self.model.generate(**inputs)
        except Exception as e:
            pass
        self.model.model.save_embedding_flag = False
