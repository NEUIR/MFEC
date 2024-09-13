import argparse
import json
import math
import random

import numpy as np
import torch
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
    parser.add_argument('--template', default='')
    args = parser.parse_args()
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    task1_data = []
    task2_data = []
    task3_data_t = []
    task3_data_f = []
    with open(args.input_path, 'r') as fin:
        for step, line in enumerate(fin):
            example = json.loads(line)
            input_claim = example['input_claim']
            span = example['span']
            evidence = example['evidence'][0]
            if args.template == 'mask':
                task1 = input_claim.replace(span, '<extra_id_0>')
                pattern_evidence_task1 = PATTERNS['mask']
                input = pattern_evidence_task1[0]
                output = pattern_evidence_task1[1]
                input_prompt_evidence_task1 = input.format(claim=task1, evidence=evidence)
                output_prompt_evidence_task1 = output.format(span=span)
                instance_evidence_task1 = {'answer': span,
                              'evidence': example['evidence'], 'input_claim': input_claim,
                              'input': input_prompt_evidence_task1, 'output': output_prompt_evidence_task1}
                task1_data.append(instance_evidence_task1)
    with open(args.output_path, 'w') as fout:
        for line in task1_data:
            fout.write(json.dumps(line) + '\n')
