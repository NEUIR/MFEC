export CUDA_VISIBLE_DEVICES=2

#FEC

#python train_rouge2.py \
#--train_data_path ../data/training_data/train_nq_base_mask.jsonl \
#--valid_data_path ../data/training_data/dev_nq_base_mask.jsonl \
#--model_name_or_path /home/users/lyq/plm/t5-base \
#--train_batch_size 8 \
#--dev_batch_size 8 \
#--gradient_accumulation_steps 4 \
#--num_train_epochs 5 \
#--output_dir ./checkpoints/FEC

##MFEC

python train_rouge2.py \
--train_data_path ../data/training_data/train_nq_generator_span.jsonl \
--valid_data_path ../data/training_data/dev_nq_generator_span.jsonl \
--model_name_or_path /home/users/lyq/plm/t5-base \
--train_batch_size 8 \
--dev_batch_size 8 \
--gradient_accumulation_steps 4 \
--num_train_epochs 5 \
--output_dir ./checkpoints/MFEC