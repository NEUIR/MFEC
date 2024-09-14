# This file is a api call about chatgpt model
# Author: Hanbin Wang
# Date: 2023/4/27

# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import argparse
import re
import backoff
import openai
import os
import json
from tqdm import tqdm
import datetime
import pickle
from tqdm import tqdm
import time
# from evaluator import smooth_bleu
# from evaluator.CodeBLEU import calc_code_bleu
# from evaluator.bleu import _bleu

import numpy as np
import re
# "https_proxy": "socks5://127.0.0.1:7890"
# os.environ["http_proxy"]="127.0.0.1:7890"
# os.environ["https_proxy"]="127.0.0.1:7890"
# os.environ["http_proxy"] ="socks5://127.0.0.1:7890" #"socks5h://localhost:10808"
# os.environ["https_proxy"] ="socks5://127.0.0.1:7890"   #"socks5h://localhost:10808"
# from openai import OpenAI
# client = OpenAI()
# import os

# os.environ["OPENAI_BASE_URL"] = "https://gtapi.xiaoerchaoren.com:8932/v1"
openai.api_base = "https://www.jcapikey.com/v1"


prompt_dict = {
    "prompt6": '''
    You will be given a piece of evidence and a claim related to the evidence. Your task is to rate the claim based on a specific criterion.
    Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.
    Evaluation Criterion:
    Relevance (1-5): The relevance between the claim and the evidence. All facts in the claim should be verifiable in the evidence.
    Evaluation Steps:
    1.Carefully read the evidence.
    2.Understand the claim and compare it with the evidence. Check if the facts in the claim can be verified by the evidence.
    3.Assign a score for relevance on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.
    Example: 
    Evidence: 
    {evidence} 
    Claim: 
    {claim}
    Evaluation Form (scores ONLY): 
    -Relevance: 
    ''',
}
def quota_giveup(e):
    return isinstance(e, openai.error.RateLimitError) and "quota" in str(e)
@backoff.on_exception(
    backoff.constant,
    openai.error.OpenAIError,
    giveup=quota_giveup,
    raise_on_giveup=True,
    interval=6
)
def chat(input):
    completion = openai.ChatCompletion.create(
        # model='gpt-4-1106-preview',
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a fact-checker."},
            {"role": "user", "content": f"{input}"},
        ]
    )
    return completion.choices[0].message

def save_response(output_path,response):
    with open(output_path, 'a',encoding='utf-8') as f:
        json.dump(response, f,ensure_ascii=False)
        f.write('\n')

def gen_csn_python(input_path,output_path,index):

    with open(input_path, 'r',encoding='utf-8') as fin:
        final_answer = []
        evidences = []
        for l in fin.readlines():
            sample = json.loads(l)
            evidences.append(' '.join(sample['evidence']))
            final_answer.append(sample['final_answer'])
        # step=0
        for step in range(len(evidences)):
            index = (index + 1) % len(apikey_list)
            openai.api_key = apikey_list[index]
            prompt_con = prompt_dict['prompt6']
            prompt_con = prompt_con.format(evidence=evidences[step], claim=final_answer[step])
            response = chat(prompt_con)
            response = response['content']
            response = response.strip('\n')
            similarity_score_match = re.search(r'Relevance[:：]\s*(\S+)', response)
            if similarity_score_match:
                similarity_score = similarity_score_match
            else:
                similarity_score = response

            item = {}
            item["evidence"] = evidences[step]
            item["claim"] = final_answer[step]
            item["similarity_score"] = str(similarity_score)
            save_response(output_path, item)



if __name__ == '__main__':

    apikey_list=['api-key',]

    index=0
    # gen_csn_python('../inference/output/mfec_fever.jsonl','./gpt/output/mfec_fever.jsonl',index)
    # gen_csn_python('../inference/output/mfec_scifact.jsonl','./gpt/output/mfec_scifact.jsonl',index)

    # gen_csn_python('../inference/output/fec_fever.jsonl','./gpt/output/fec_fever.jsonl',index)
    # gen_csn_python('../inference/output/fec_scifact.jsonl','./gpt/output/fec_scifact.jsonl',index)

    # gen_csn_python('../inference/output/ft_fever.jsonl','./gpt/output/ft_fever.jsonl',index)
    # gen_csn_python('../inference/output/ft_scifact.jsonl','./gpt/output/ft_scifact.jsonl',index)

    # gen_csn_python('../inference/output/zerofec_fever.json','./gpt/output/zerofec_fever.json',index)
    # gen_csn_python('../inference/output/zerofec_scifact.json','./gpt/output/zerofec_scifact.json',index)

    # gen_csn_python('../inference/output/zeroshot_fever.jsonl','./gpt/output/zeroshot_fever.jsonl',index)
    # gen_csn_python('../inference/output/zeroshot_scifact.jsonl','./gpt/output/zeroshot_scifact.jsonl',index)

    # gen_csn_python('../inference/output/compedit_fever.jsonl','./gpt/output/compedit_fever.jsonl',index)
    # gen_csn_python('../inference/output/compedit_scifact.jsonl','./gpt/output/compedit_scifact.jsonl',index)

