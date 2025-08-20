from datasets import Dataset, load_from_disk
import json
from PIL import Image
from IPython.display import display
from io import BytesIO
import pandas as pd
from vlmeval.smp.vlm import encode_image_to_base64
from ..smp import *
import os

class CustomDataset:
    TYPE = "MIX"
    MODALITY = "MIX"
    names = [x for x in os.listdir(os.environ["LUMINA"]+"/Datasets/") if os.path.isfile(os.environ["LUMINA"]+"/Datasets/"+x+"/dataset_info.json")]
    is_raw = False
    def __init__(self, dataset="CustomDataset", **kwargs):
        ROOT = LMUDataRoot()
        self.idx = 0
        # You can override this variable to save image files to a different directory
        if "QUERY_FILE" not in os.environ:
            self.names = [x for x in os.listdir(os.environ["LUMINA"]+"/Datasets/") if os.path.isfile(os.environ["LUMINA"]+"/Datasets/"+x+"/dataset_info.json")]
            self.names.sort()
            print("Select index [0, "+str(len(self.names)-1)+"] from avaliable custom datasets:")
            for i in range(len(self.names)):
                print("\t"+str(i)+") "+self.names[i])
            idx = int(input("idx:"))
            self.name = self.names[idx]
        else:
            self.name = self.names[0]
        self.dataset_name = dataset + "_" + self.name
        os.environ["DATASET_NAME"] = self.dataset_name
        self.img_root = osp.join(ROOT, 'images', self.dataset_name)
        if not os.path.exists(self.img_root):
            os.makedirs(self.img_root)
        file = f"{os.environ['LUMINA']}/Datasets/" + self.name
        self.dataset = load_from_disk(file)
        try:
            metadata = json.loads(self.dataset.info.description)
            if "mode" in metadata and metadata["mode"] == "raw":
                self.is_raw = True
                print("Dataset mode set to raw.")
        except:
            pass
        self.data = pd.DataFrame(self.dataset)
        self.n = len(self.data["question"])
        self.data["index"] = [i for i in range(self.n)]
        self.data["category"] = ["0" for i in range(self.n)]
        if True and "QUERY_FILE" in os.environ:
            self.data = pd.DataFrame({"index":[0],"category":[0],"question":"?"})
            self.n = 1
    def __len__(self):
        print("Dataset size:", self.n)
        return self.n

    def create_text_query(self, el):
        if len(el["choices"]) != 0:
            txt = f"""Choose the option that best answers the folowing question. Don't justify or explain the answer. Answer with a single character: A, B, C or D.
Question: {el['question']}
Options:\nA: {el['choices'][0]}\nB: '{el['choices'][1]}'"""
            if len(el['choices']) > 2:
                txt += f"\nC: '{el['choices'][2]}'"
            if len(el['choices']) > 3:
                txt += f"\nD: '{el['choices'][3]}'"
            return txt
        else:
            txt = f"""{el['question']}"""
            return txt
    
    def build_prompt(self, line):
        res = []
        if "QUERY_FILE" not in os.environ:
            question = self.create_text_query(line)
            images = line["image"]
            print("NUMBER OF IMAGES:",len(images))
            image_paths = []
            for idx, img in enumerate(images):
                Image.open(BytesIO(img["bytes"])).save(osp.join(self.img_root, str(line["index"]) + "_" + str(idx) + ".png"))
                image_paths.append(osp.join(self.img_root, str(line["index"]) + "_" + str(idx) + ".png"))
            if len(images):
                question = f"You are a helpful asistant that can understand the images provided by the User and answer the questions asked.\nUser: {"<image_placeholder>"*len(images)}\n{question}\nAssistant:"
            else:
                question = f"You are a helpful asistant that can understand the input provided by the User and answer the questions asked.\nUser: {question}\nAssistant:"
            if self.is_raw:
                question = ("<image_placeholder>"*len(images)) + line["question"]
            parts = question.split("<image_placeholder>")
            assert len(parts) == len(images)+1
            for i in range(len(image_paths)):
                if len(parts[i])>0:
                    res.append(dict(type='text', value=parts[i]))
                res.append(dict(type='image', value=image_paths[i]))
            if len(parts[-1])>0:
                res.append(dict(type='text', value=parts[-1]))
        else:
            file = os.environ["QUERY_FILE"]
            with open(file, "r") as f:
                res = [dict(type="text", value=f.read())]
        self.idx += 1
        os.environ["CURRENT_DATASET_INDEX"] = str(self.idx)
        return res
        
    def evaluate(self, eval_file, **judge_kwargs):
        print("Evaluation not implemented.")
    def dump_image(self, line):
        return None
    def supported_datasets(self):
        base = "CustomDataset"
        res = [base]
        for name in self.names:
            res.append(base + "_" + name)
        return res
    @classmethod
    def supported_datasets(self):
        base = "CustomDataset"
        res = [base]
        for name in self.names:
            res.append(base + "_" + name)
        return res
