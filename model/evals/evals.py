import re
import string

import evaluate

from bart_score import BARTScorer
import argparse
import json
import numpy as np
import rouge
from transformers import BertForSequenceClassification, BertTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--predicted_path', required=True)
parser.add_argument('--gold_path', required=True)
parser.add_argument('--eval_evidence', action='store_true')

args = parser.parse_args()

with open(args.predicted_path,'r') as f:

    predictions = []
    inputs = []
    for l in f.readlines():
        sample = json.loads(l)
        if len(sample['evidence'])==0:
            continue
        inputs.append(sample['input_claim'])
        predictions.append(sample['final_answer'])
    print(len(inputs))

with open(args.gold_path, 'r') as f:
    gts = []
    sari_gts = []
    evidences = []
    for l in f.readlines():
        sample = json.loads(l)
        if len(sample['evidence'])==0:
            continue
        gts.append(sample['gt_claim'])
        sari_gts.append([sample['gt_claim']])
        evidences.append(' '.join(sample['evidence']))
        # print(evidences)
    print(len(gts))
bart_scorer = BARTScorer(device='cuda:0', checkpoint='../../../plm/bart_score')
bart_scores = bart_scorer.score(evidences, predictions, batch_size=4)
#
print("BART Score cnn", np.mean(bart_scores))

# print("BART Score trained", np.mean(bart_scores))
sari = evaluate.load("./sari_evaluate.py")
sari_score = sari.compute(sources=inputs, predictions=predictions, references=sari_gts)
print(sari_score)
rouge=evaluate.load('./rouge.py')
rouge_scores = rouge.compute(references=gts,predictions=predictions)
# rouge_scores = compute_rouge(predictions, gts)
print(rouge_scores)
