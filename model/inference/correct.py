import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from typing import Dict

PATTERNS = {
    "mask":[
        ("Please determine the content that should be inserted into the masked position of the claim based on the provided evidence and output the filled span:\nClaim: {claim}\nEvidence: {evidence}",
            "{span}")],
    "mfec":[
        ("Please evaluate the accuracy of the highlighted span in the claim using the provided evidence, and provide a corrected version of the span:\nClaim: {claim}\nEvidence: {evidence}",
            "{span}")]

}
class CorrectGenerator:
    def __init__(self, args):
        self.args = args
        model_name_or_path = args.model_name_path
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.task = args.task

    def generate_span(self, input_list):
        input_ids = self.tokenizer.batch_encode_plus(input_list, return_tensors="pt", padding='longest', truncation=True,max_length=512).input_ids.cuda()
        with torch.no_grad():
            self.model.eval()
            generated_ids = self.model.generate(input_ids, max_length=20, num_beams=4, early_stopping=True)
        candidate_span = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return candidate_span

    def generate_candidate(self, sample: Dict):
        input_claim = sample['input_claim']
        candidate_answers = sample['candidate_answers']
        evidence = ''
        for evi in sample['evidence']:
            evidence = evidence + evi
        sample['corrections'] = []
        sample['generate_span'] = []

        input_list = []
        for span in candidate_answers:
            if span != '' and span != ' ':
                if self.task=='mfec':
                    for pattern in PATTERNS['mfec']:
                        input = pattern[0]
                        claim =input_claim.replace(span, '<extra_id_1> ' + span + ' <extra_id_2> ')
                        input_prompt_task1 = input.format(claim=claim, evidence=evidence)
                        input_list.append(input_prompt_task1)
                if self.task=='mask':
                    for pattern in PATTERNS['mask']:
                        input = pattern[0]
                        claim =input_claim.replace(span, '<extra_id_0>')
                        input_prompt_task2 = input.format(claim=claim, evidence=evidence)
                        input_list.append(input_prompt_task2)
        generate_spans = self.generate_span(input_list)
        sample['generate_span'].extend(generate_spans)
        index=0
        for span in candidate_answers:
            if span == '' or span == ' ':
                continue
            correction = input_claim.replace(span, generate_spans[index])
            index = index + 1
            sample['corrections'].append(correction)
        return sample