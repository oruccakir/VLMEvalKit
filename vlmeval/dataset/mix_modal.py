from .image_mcq import *
from ..smp import *


class MixModalDataSet(ImageMCQDataset):
    TYPE = 'MIX_MODAL'
    DATASET_URL = {
        "MixModalBench":'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_EN.tsv'
    }
    def __init__(self, dataset='MixModalBench', skip_noimg=True):
        print("Mix modal dataset init")
        super().__init__(dataset, skip_noimg)

    def prepare_tsv(self, url, file_md5=None):
        print("Mix modal data preapre")
        return super().prepare_tsv(url,file_md5=None)
    
    def build_prompt(self, line):
        print("Mix modal build prompt method")
        return super().build_prompt(line)

    def evaluate(self, eval_file, **judge_kwargs):
        print("Mix modal evaluate")
        return super().evaluate(eval_file, **judge_kwargs)



