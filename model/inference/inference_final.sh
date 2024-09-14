export CUDA_VISIBLE_DEVICES=2

##mfec
#python inference.py \
#--input_path ../../data/test/fever_test_clean.jsonl \
#--output_path  ./output/fever_mfec.jsonl \
#--model_name_path /home/users/lyq/MFEC/MFEC/checkpoints/mfec \
#--task mfec \

python inference.py \
--input_path ../../data/test/scifact_test_clean.jsonl \
--output_path  ./output/scifact_mfec.jsonl \
--model_name_path /home/users/lyq/MFEC/MFEC/checkpoints/mfec \
--task mfec \
--use_scispacy

##FEC
#python inference.py \
#--input_path ../../data/test/fever_test_clean.jsonl \
#--output_path  ./output/fever_fec.jsonl \
#--model_name_path /home/users/lyq/MFEC/MFEC/checkpoints/FEC \
#--task mask \

python inference.py \
--input_path ../../data/test/scifact_test_clean.jsonl \
--output_path  ./output/scifact_fec.jsonl \
--model_name_path /home/users/lyq/MFEC/MFEC/checkpoints/FEC \
--task mask \
--use_scispacy

##zeroshot
#python inference.py \
#--input_path ../../data/test/fever_test_clean.jsonl \
#--output_path  ./output/fever_zeroshot.jsonl \
#--model_name_path t5-base \
#--task mask \

#python inference.py \
#--input_path ../../data/test/scifact_test_clean.jsonl \
#--output_path  ./output/scifact_zeroshot.jsonl \
#--model_name_path t5-base \
#--task mask \
#--use_scispacy

