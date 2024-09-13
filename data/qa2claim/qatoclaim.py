import argparse
import json
import math
from typing import List, Dict

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def format_inputs(question: str, answer: str):
    return f"{answer} \\n {question}"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='./nq-dev-kilt.jsonl')
    parser.add_argument('--output_path', default='./nq-dev-kilt_outputs.jsonl')
    parser.add_argument('--qa2claim_model_path', default='../../../plm/qa2claim')
    parser.add_argument('--dataset')
    args = parser.parse_args()
    qa2s_tokenizer_path = args.qa2claim_model_path
    qa2s_model_path = args.qa2claim_model_path
    model = AutoModelForSeq2SeqLM.from_pretrained(qa2s_model_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(qa2s_tokenizer_path)
    input_path_list = ['../kilt/nq-dev-kilt_outputs.jsonl', '../kilt/nq-train-kilt_outputs.jsonl','../kilt/triviaqa-dev-kilt_outputs.jsonl', '../kilt/triviaqa-train-kilt_outputs.jsonl']
    output_path_list = ['./nq-dev-kilt_claim_outputs.jsonl', './nq-train-kilt_claim_outputs.jsonl','./triviaqa-dev-kilt_claim_outputs.jsonl',
                        './triviaqa-train-kilt_claim_outputs.jsonl']
    for i in range(len(input_path_list)):
        output_list = []
        with open(input_path_list[i], 'r') as f:
            input_list = []
            batch_size = 256
            for l in tqdm(f):
                data = json.loads(l)
                questions = data['generated_question']
                answers = data['answer']
                input_text = format_inputs(questions, answers)
                input_list.append(input_text)
            print(len(input_list))
            index = 0
            claim_list = []
            total_step = math.ceil(len(input_list)/batch_size)
            for step in tqdm(range(total_step)):
                if index < len(input_list):
                    if index==step-1:
                        inputs = input_list[index * batch_size:]
                    else:
                        inputs = input_list[index * batch_size: (index + 1) * batch_size]
                    index = index + 1
                    input_ids = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding='longest', truncation=True,
                                                       max_length=512).input_ids.cuda()
                    generated_ids = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
                    generated_claim = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    claim_list.extend(generated_claim)
            print(len(claim_list))
        with open(input_path_list[i], 'r') as f:
            for step, l in enumerate(f):
                data = json.loads(l)
                generated_questions = data['generated_question']
                generated_answers = data['answer']
                evidence = ''
                for evi in data['evidence']:
                    evidence = evidence + evi.strip('\n')
                instance = {'answer': data['answer'], 'generated_question': data['generated_question'],
                            'evidence': [evidence], 'input_claim': claim_list[step]}
                output_list.append(instance)
        with open(output_path_list[i], 'w') as fout:
            for data in output_list:
                fout.write(json.dumps(data) + '\n')

