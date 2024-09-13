export CUDA_VISIBLE_DEVICES=7

python main_answer.py \
--input_path ../qa2claim/nq-dev-kilt_claim_outputs.jsonl \
--output_path  ./nq_dev_all_answer.jsonl \
--seed 1234

python main_answer.py \
--input_path ../qa2claim/nq-train-kilt_claim_outputs.jsonl \
--output_path  ./nq_train_all_answer.jsonl \
--seed 1234



