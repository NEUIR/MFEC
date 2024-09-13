import json
import math
import os

import numpy as np
import spacy
import torch
from sense2vec import Sense2VecComponent
from pathlib import Path
import tqdm
import json
import random
import argparse
from transformers import T5ForConditionalGeneration, AutoTokenizer

def generate(input_list,tokenizer,model,batch_size):
    span_list = []
    total_step = math.ceil(len(input_list) / batch_size)
    print(total_step)
    for step in tqdm.tqdm(range(total_step)):
        inputs = input_list[step * batch_size: (step + 1) * batch_size]
        input_ids = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding='longest',
                                                truncation=True, max_length=512).input_ids.cuda()
        with torch.no_grad():
            model.eval()
            generated_ids = model.generate(input_ids, max_length=20, num_beams=4, early_stopping=True)
            generated_span = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            span_list.extend(generated_span)
    return span_list

def process_1(inpath,tokenizer,model,batch_size,outpath=None,outpath_e=None):
    file = open(inpath, 'r')
    lines = file.readlines()
    output_list_e = []
    input_list_e = []
    process_data=[]
    for step, line in tqdm.tqdm(enumerate(lines),total=len(lines)):
        data = json.loads(line)
        input_claim = data['input_claim']
        if 'span'in data:
            span = data['span']
            evidence = data['evidence'][0]
            task2 = input_claim.replace(span, '<extra_id_0>')
            pattern_e = PATTERNS['mask']
            input = pattern_e[0]
            input_prompt_task1 = input.format(claim=task2, evidence=evidence)
            input_list_e.append(input_prompt_task1)
            process_data.append(data)
        else:
            print(data)
            continue
    span_list_e = generate(input_list_e,tokenizer,model,batch_size)
    true=[]
    false=[]
    for step,data in enumerate(process_data):
        input_claim = data['input_claim']
        if 'span' in data:
            span = data['span']
            evidence = data['evidence'][0]

            task_span = input_claim.replace(span, '<extra_id_1> ' + span_list_e[step] + ' <extra_id_2> ')
            pattern_task3 = PATTERNS['mfec']
            input = pattern_task3[0]
            output = pattern_task3[1]
            input_prompt_evidence_task3 = input.format(claim=task_span, evidence=evidence)
            output_prompt_evidence_task3 = output.format(span=span)
            instance_evidence_task3 = {'answer': span, 'reverse_answer': span_list_e[step],
                                       'evidence': data['evidence'], 'input_claim': input_claim,
                                       'gt_claim': data['input_claim'],
                                       'input': input_prompt_evidence_task3, 'output': output_prompt_evidence_task3,
                                       'task_id': 1}
            if span == span_list_e[step]:
                true.append(instance_evidence_task3)
            else:
                false.append(instance_evidence_task3)
    true_len=len(true)
    false_len=len(false)
    if true_len<false_len:
        length=true_len
    else:
        length=false_len
    random.shuffle(true)
    random.shuffle(false)
    output_list_e.extend(true[:length])
    output_list_e.extend(false[:length])
    random.shuffle(output_list_e)
    with open(outpath_e, 'w') as fout:
            for data in tqdm.tqdm(output_list_e, total=len(output_list_e)):
                fout.write(json.dumps(data) + '\n')
if __name__ == '__main__':
    PATTERNS = {
        "mask":
            (
                "Please determine the content that should be inserted into the masked position of the claim based on the provided evidence and output the filled span:\nClaim: {claim}\nEvidence: {evidence}",
                "{span}"),
        "mfec":
            (
                "Please evaluate the accuracy of the highlighted span in the claim using the provided evidence, and provide a corrected version of the span:\nClaim: {claim}\nEvidence: {evidence}",
                "{span}")

    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='')
    parser.add_argument('--output_path', default='')
    parser.add_argument('--output_path_e', default='')
    parser.add_argument('--model_name_or_path', default='')
    parser.add_argument('--batch_size', type=int,default=8)
    args = parser.parse_args()
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    process_1(args.input_path,tokenizer,model,args.batch_size,args.output_path,args.output_path_e)

