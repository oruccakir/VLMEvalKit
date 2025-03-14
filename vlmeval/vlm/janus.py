import sys
import torch
from transformers import AutoModelForCausalLM, AutoConfig,BitsAndBytesConfig
import warnings
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE

from imports import DEEPSEEK_MODEL_EMBEDDINS_DIR_PATH

class Janus(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def check_install(self):
        try:
            import janus
        except Exception as e:
            logging.critical(
                'Please first install janus from source codes in: https://github.com/deepseek-ai/Janus')
            raise e

    def __init__(self, model_path='deepseek-ai/Janus-1.3B', **kwargs):
        self.check_install()
        assert model_path is not None
        self.model_path = model_path
        from janus.models import VLChatProcessor

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

        self.vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map=self.device_map,torch_dtype=torch.bfloat16, quantization_config=qunatization_config) if apply_quantization else AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map=self.device_map,torch_dtype=torch.bfloat16)
        self.model = model

        self.save_embeddings = kwargs["config"]["save_embedding_flag"] if "config" in kwargs else False

        self.idx = 0

        if "apply_quantization" in kwargs:
            del kwargs["apply_quantization"]
        if "device_map" in kwargs:
            del kwargs['device_map']
        if "quant_config" in kwargs:
            del kwargs['quant_config']
        if "config" in kwargs:
            del kwargs['config']

        torch.cuda.empty_cache()
        default_kwargs = dict(
            max_new_tokens=2048,
            do_sample=False,
            use_cache=True,
            output_logits=False,
            output_scores=False,
            return_dict_in_generate=False)

        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def prepare_inputs(self, message):
        def prepare_itlist(msgs):
            content, images = '', []
            for s in msgs:
                if s['type'] == 'image':
                    images.append(s['value'])
                    content += '<image_placeholder>'
                elif s['type'] == 'text':
                    content += s['value']
            return content, images
        conversation = []
        if 'role' not in message[0]:
            content, images = prepare_itlist(message)
            conversation.append(dict(role='User', content=content, images=images))
        else:
            role_map = {'user': 'User', 'assistant': 'Assistant'}
            for msgs in message:
                role = role_map[msgs['role']]
                content, images = prepare_itlist(msgs['content'])
                conversation.append(dict(role=role, content=content, images=images))
        conversation.append(dict(role='Assistant', content=''))
        return conversation

    def generate_inner(self, message, dataset=None,category=None):
        if dataset is None or not ('MMVet' in dataset):
            self.vl_chat_processor.system_prompt = ""
        else:
            self.vl_chat_processor.system_prompt = "You are a helpful assistant. Please answer truthfully and write out your thinking step by step to be sure you get the right answer."  # noqa: E501

        conversation = self.prepare_inputs(message)
        from janus.utils.io import load_pil_images
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(conversations=conversation, images=pil_images, force_batchify=True)
        prepare_inputs = prepare_inputs.to(self.model.device, dtype=torch.bfloat16)
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)


        if self.save_embeddings:
            embedd_dir_path=f"{DEEPSEEK_MODEL_EMBEDDINS_DIR_PATH}/{dataset}"
            if not os.path.exists(embedd_dir_path):
                os.makedirs(embedd_dir_path)
            
            embedding_file_path = f"{embedd_dir_path}/embedding_{self.idx}.bin"
            self.idx += 1

            embds = inputs_embeds
            embds.cpu().flatten().float().detach().numpy().tofile(embedding_file_path)
            print(f"Embeddings saved to {embedding_file_path} with {embds.shape} tokens")

        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **self.kwargs)
        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer

    def chat_inner(self, message, dataset=None):
        return self.generate_inner(message, dataset=dataset)

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'Y/N' or DATASET_TYPE(dataset) == 'MCQ' or dataset == 'MMVet':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        if DATASET_TYPE(dataset) == 'Y/N':
            if dataset == 'POPE':
                question = question.replace(" Please answer yes or no.", "")
            prompt = '\n' + question + "\nAnswer the question using a single word or phrase."
        elif DATASET_TYPE(dataset) == 'MCQ':
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
                f"{options_prompt}\nAnswer with the option's letter from the given choices directly."
                if len(options) else 'Answer the question directly. '
            )
        elif dataset == 'MMVet':
            prompt = '\n' + question
        else:
            raise NotImplementedError

        message = [dict(type='image', value=s) for s in tgt_path]
        message.extend([dict(type='text', value=prompt)])
        return message
