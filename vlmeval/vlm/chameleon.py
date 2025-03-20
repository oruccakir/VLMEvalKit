import os.path as osp
import warnings
from .base import BaseModel
from ..smp import *
from PIL import Image
import torch

from ..dataset import DATASET_TYPE, DATASET_MODALITY

from transformers import BitsAndBytesConfig
from imports import CHAMELEON_MODEL_EMBEDDINS_DIR_PATH, CHAMELEON_MODEL_HF_DIR_PATH

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

        processor = ChameleonProcessor.from_pretrained(model_path,cache_dir=CHAMELEON_MODEL_HF_DIR_PATH)
        model = ChameleonForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=self.device_map, quantization_config=qunatization_config,cache_dir=CHAMELEON_MODEL_HF_DIR_PATH) if apply_quantization else ChameleonForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=self.device_map,cache_dir=CHAMELEON_MODEL_HF_DIR_PATH)

        self.save_embeddings = kwargs["config"]["save_embedding_flag"] if "config" in kwargs else False
        self.save_embeddings_by_category = kwargs["config"]["save_embedding_by_category_flag"] if "config" in kwargs else False
        self.prev_category = None
        self.number_of_embeddings_per_ctg = kwargs["config"]["number_of_embeddings_for_each_category"] if "config" in kwargs else 1

        self.model = model
        self.processor = processor

        if "config" in kwargs:
            del kwargs['config']

        self.idx = 0

    def generate_inner(self, message, dataset=None,category=None):
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
        ).to(device=self.device_map, dtype=torch.bfloat16) if len(images) > 0 else self.processor(
            text=[content],
            padding=True,
            return_tensors='pt'
        ).to(device=self.device_map, dtype=torch.bfloat16)

        #print("Builded prompt")
        #print(content)

        if self.save_embeddings:
            embedd_dir_path=f"{CHAMELEON_MODEL_EMBEDDINS_DIR_PATH}/{dataset}"
            if not os.path.exists(embedd_dir_path):
                os.makedirs(embedd_dir_path)
            
            if self.save_embeddings_by_category == False:
                embedding_file_path = f"{embedd_dir_path}/embedding_{self.idx}.bin"
                self.idx += 1

                self.compute_and_save_embeddings(inputs,embedding_file_path)
            else:
                if category != self.prev_category:
                    embedding_file_path = f"{embedd_dir_path}/embedding_{category.lower().replace(' ', '_')}_{self.idx}.bin"
                    self.idx = self.idx + 1
                    self.compute_and_save_embeddings(inputs,embedding_file_path)
                    if self.idx == self.number_of_embeddings_per_ctg:   
                        self.prev_category = category
                        self.idx = 0


        generate_ids = self.model.generate(**inputs, max_new_tokens=2048)
        input_token_len = inputs.input_ids.shape[1]
        text = self.processor.batch_decode(
            generate_ids[:, input_token_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        #print("Generated text")

        return text


    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == "MCQ":
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        if DATASET_TYPE(dataset) == 'MCQ':
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = ''
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'

            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            prompt = f'\nHint: {hint}\n' if hint is not None else '\n'
            prompt += f'{question}\n'
            prompt += (
                f"{options_prompt}\nAnswer with the option's letter from the given choices directly.Always give the answer in the form of just single letter."
                if len(options) else 'Answer the question directly. just the letter.Always give the answer in the form of just single letter.'
            )
        else:
            raise NotImplementedError

        message = [dict(type='image', value=s) for s in tgt_path]
        message.extend([dict(type='text', value=prompt)])
        return message
    
    def compute_and_save_embeddings(self, inputs, embedding_file_path):
        self.model.model.save_embedding_flag = True
        self.model.model.embedding_file_path = embedding_file_path
        try :
            self.model.generate(**inputs)
        except Exception as e:
            pass
        self.model.model.save_embedding_flag = False
