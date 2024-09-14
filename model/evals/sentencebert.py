import argparse
import json
import os

import torch
from sentence_transformers import SentenceTransformer, models
model_path = 'roberta-large-nli-stsb-mean-tokens'
model = SentenceTransformer(model_name_or_path=model_path)
def calculate_sentencebert(input_path):
    with open(input_path, 'r') as f:
        sentence_bert=0
        count=0
        for l in f.readlines():
            sample = json.loads(l)
            gt_claim = sample['gt_claim']
            final_answer=sample['final_answer']
            if len(sample['evidence']) == 0:
                continue
            embeddings = model.encode([gt_claim,final_answer])
            similarities = model.similarity(embeddings, embeddings)
            sentence_bert = sentence_bert + similarities[0][1]
            count = count + 1
        sentence_bert = sentence_bert/count
        print(count)
        return sentence_bert
        print('final EM_seq',sentence_bert)
if __name__ == '__main__':
    sentence_bert=calculate_sentencebert('../inference/output/compedit_fever.jsonl')
    print('final sentence_bert of compedit_fever',sentence_bert)
    sentence_bert = calculate_sentencebert('../inference/output/compedit_scifact.jsonl')
    print('final sentence_bert of compedit_sci', sentence_bert)
    sentence_bert = calculate_sentencebert('../inference/output/zerofec_fever.json')
    print('final sentence_bert of zerofec_fever', sentence_bert)
    sentence_bert = calculate_sentencebert('../inference/output/zerofec_scifact.json')
    print('final sentence_bert of zerofec_sci', sentence_bert)
    sentence_bert = calculate_sentencebert('../inference/output/selfask_nq_noevidence_base_span_rouge2_same_clean.jsonl')
    print('final sentence_bert of mfec-fever', sentence_bert)
    sentence_bert = calculate_sentencebert('../inference/output/scifact_selfask_nq_noevidence_base_span_rouge2_same_clean.jsonl')
    print('final sentence_bert of mfec-sci', sentence_bert)
    sentence_bert = calculate_sentencebert('../inference/output/ft_fever.jsonl')
    print('final sentence_bert of ft-fever', sentence_bert)
    sentence_bert = calculate_sentencebert('../inference/output/ft_scifact.jsonl')
    print('final sentence_bert of ft-sci', sentence_bert)
    sentence_bert = calculate_sentencebert('../inference/output/zeroshot_fever.jsonl')
    print('final sentence_bert of zeroshot-fever', sentence_bert)
    sentence_bert = calculate_sentencebert('../inference/output/zeroshot_scifact.jsonl')
    print('final sentence_bert of zeroshot-sci', sentence_bert)
    sentence_bert = calculate_sentencebert('../inference/output/fec_fever.jsonl')
    print('final sentence_bert of fec_fever', sentence_bert)
    sentence_bert = calculate_sentencebert('../inference/output/fec_scifact.jsonl')
    print('final sentence_bert of  fec-sci', sentence_bert)


