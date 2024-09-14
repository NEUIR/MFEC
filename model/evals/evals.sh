export CUDA_VISIBLE_DEVICES=7
#zerofec
python evals.py \
--predicted_path  ../inference/output/zeroshot_scifact.jsonl \
--gold_path  ../inference/output/zeroshot_scifact.jsonl \

python evals.py \
--predicted_path  ../inference/output/zeroshot_fever.jsonl \
--gold_path  ../inference/output/zeroshot_fever.jsonl \
###mfec
python evals.py \
--predicted_path  ../inference/output/mfec_scifact.jsonl \
--gold_path  ../inference/output/mfec_scifact.jsonl \

python evals.py \
--predicted_path  ../inference/output/mfec_fever.jsonl \
--gold_path  ../inference/output/mfec_fever.jsonl \


##fec
python evals.py \
--predicted_path  ../inference/output/fec_scifact.jsonl \
--gold_path  ../inference/output/fec_scifact.jsonl \

python evals.py \
--predicted_path  ../inference/output/fec_fever.jsonl \
--gold_path  ../inference/output/fec_fever.jsonl \

##ft
python evals.py \
--predicted_path  ../inference/output/ft_scifact.jsonl \
--gold_path  ../inference/output/ft_scifact.jsonl \

python evals.py \
--predicted_path  ../inference/output/ft_fever.jsonl \
--gold_path  ../inference/output/ft_fever.jsonl \

##zerofec
python evals.py \
--predicted_path  ../inference/output/zerofec_scifact.json \
--gold_path  ../inference/output/zerofec_scifact.json \

python evals.py \
--predicted_path  ../inference/output/zerofec_fever.json \
--gold_path  ../inference/output/zerofec_fever.json \

##compedit
python evals.py \
--predicted_path  ../inference/output/compedit_scifact.jsonl \
--gold_path  ../inference/output/compedit_scifact.jsonl \

python evals.py \
--predicted_path  ../inference/output/compedit_fever.jsonl \
--gold_path  ../inference/output/compedit_fever.jsonl \
