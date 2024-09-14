from model import INFERMODEL
from types import SimpleNamespace
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', required=True)
parser.add_argument('--output_path', required=True)
parser.add_argument('--model_name_path', required=True)
parser.add_argument('--entailment_model_path', default='../../../plm/domain_docnli/docnli-roberta_pubmed_bioasq.pt')
parser.add_argument('--entailment_tokenizer_path', default='../../../plm/roberta_large')
parser.add_argument("--use_scispacy",default=False,action="store_true")
parser.add_argument("--task",default='')
args = parser.parse_args()


inference_model = INFERMODEL(args)

with open(args.input_path,'r') as f:
    inputs = [json.loads(l) for l in f.readlines()]
print('success load data')
outputs = inference_model.batch_correct(inputs)

with open(args.output_path,'w') as f:
    for output in outputs:
        f.write(json.dumps(output)+'\n')