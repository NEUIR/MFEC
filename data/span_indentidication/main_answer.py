import random
from tqdm import tqdm
from typing import List, Dict
import numpy as np
import torch
import argparse
import json
from answer_selector import AnswerSelector

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', required=True)
parser.add_argument('--output_path', required=True)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--dataset')
args = parser.parse_args()
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
answer_selector = AnswerSelector()
def correct(sample: Dict):
    '''
    sample is Dict containing at least two fields:
        inputs: str, the claim to be corrected.
        evidence: str, the list of reference article to check against.
    '''
    sample = answer_selector.select_answers(sample)
    return sample
def batch_correct(samples: List[Dict]):
    return [correct(sample) for sample in tqdm(samples, total=len(samples))]
set_seed(args)
with open(args.input_path,'r') as f:
    inputs = [json.loads(l) for l in f.readlines()]
print('success load data')
outputs = batch_correct(inputs)
with open(args.output_path,'w') as f:
    for output in outputs:
        f.write(json.dumps(output)+'\n')