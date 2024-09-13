export CUDA_VISIBLE_DEVICES=1
export TRANSFORMERS_CACHE=/home/users/lyq/.cache


#python process_pretrain_fec_getclaim_mask.py \
#--input_path ../span_indentidication/nq_dev_all_answer.jsonl \
#--output_path ./dev_nq_base_mask.jsonl \
#--template mask
#
#
#python process_pretrain_fec_getclaim_mask.py \
#--input_path ../span_indentidication/nq_train_all_answer.jsonl \
#--output_path ./train_nq_base_mask.jsonl \
#--template mask

python onlymask_generator.py \
--input_path ../span_indentidication/nq_dev_all_answer.jsonl \
--output_path_e ./dev_nq_generator_span.jsonl \
--model_name_or_path /home/users/lyq/MFEC/MFEC/checkpoints/FEC  \
--batch_size 32

python onlymask_generator.py \
--input_path ../span_indentidication/nq_train_all_answer.jsonl \
--output_path_e ./train_nq_generator_span.jsonl \
--model_name_or_path ../MFEC/checkpoints/FEC  \
--batch_size 32
