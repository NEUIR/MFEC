import sys
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    AutoModel, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, T5ForConditionalGeneration
)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import logging
import random
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, DistributedSampler
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import load_metric,load_dataset
import evaluate
print(evaluate.load('./rouge.py').compute(references=['hello'], predictions=['hello']))
import nltk
logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def load_stuff(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # model = INSTRUCTModel(args)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model.cuda()
    return tokenizer, model

def get_arguments():
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument(
        "--train_data_path",
        default=None,
        type=str,
        required=True,
        help="The training data path.",
    )

    parser.add_argument(
        "--valid_data_path",
        default=None,
        type=str,
        required=True,
        help="The validation data path.",
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        # required=True,
        help="The checkpoint data path.",
    )

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )


    parser.add_argument(
        "--max_len",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )


    parser.add_argument(
        "--train_batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )

    parser.add_argument(
        "--dev_batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )

    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.",
    )

    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.",
    )

    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=5.0,
        type=float,
        help="Total number of training epochs to perform.",
    )

    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps.",
    )
    parser.add_argument('--patience', type=int, default=5, help='Patience')

    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1000,
        help="eval model every X updates steps.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="eval model every X updates steps.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=5,
        help="eval model every X updates steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="random seed for initialization",
    )
    args = parser.parse_args()

    return args


def set_env(args):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    handlers = [logging.FileHandler(os.path.abspath(args.output_dir) + '/train_log.txt'), logging.StreamHandler()]
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers = handlers)
    # Set seed
    set_seed(args)





def main():
    args = get_arguments()
    set_env(args)
    logger.info("Training/evaluation parameters %s", args)
    tokenizer, model = load_stuff(args)

    logger.info("Loading training set.")
    print(args.train_data_path)
    data_files = {"train": args.train_data_path, "validation": args.valid_data_path}
    dataset = load_dataset('json', data_files=data_files)

    def preprocess_function(examples):
        inputs = [doc for doc in examples["input"]]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True)

        labels = tokenizer(text_target=examples["output"], max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset = dataset.map(preprocess_function, batched=True)
    #
    # logger.info("Loading validation set.")
    # valid_dataset = load_dataset('json', data_files=args.valid_data_path,split="validation")
    # valid_dataset = valid_dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.dev_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=0.01,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        fp16=False,
        predict_with_generate=True,
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model='eval_rouge2',
        save_strategy='steps',
        save_steps=args.save_steps,
        logging_dir=f"{args.output_dir}/logs",
        log_level='info',
        logging_strategy='steps',
        logging_steps=10,
        seed=args.seed,
        data_seed=args.seed,
    )

    nltk.download("punkt", quiet=True)
    def compute_metrics(eval_preds):
        metric = evaluate.load('./rouge.py')
        preds, labels = eval_preds

        # decode preds and labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # rougeLSum expects newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return result
    train_dataset = dataset['train']
    eval_dataset = dataset['validation']
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    print('start training')
    trainer.train()



if __name__ == "__main__":
    main()