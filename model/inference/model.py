from answer_selector import AnswerSelector
from entailment_model import EntailmentModel
from correct import CorrectGenerator
from tqdm import tqdm
from typing import List, Dict

import nltk

# nltk.download('punkt')


class INFERMODEL:

    def __init__(self, args) -> None:
        # init all the model
        self.args = args
        self.answer_selector = AnswerSelector(args)
        self.correction = CorrectGenerator(args)
        self.entailment_model = EntailmentModel(args)
        print("Finish loading models.")

    def correct(self, sample: Dict):
        '''
        sample is Dict containing at least two fields:
            inputs: str, the claim to be corrected.
            evidence: str, the list of reference article to check against.
        '''

        sample = self.answer_selector.select_answers(sample)
        sample = self.correction.generate_candidate(sample)
        sample = self.entailment_model.run_entailment_prediction(sample)
        return sample
    def batch_correct(self, samples: List[Dict]):
        return [self.correct(sample) for sample in tqdm(samples, total=len(samples))]
