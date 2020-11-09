data_dir=${1:-090820_bootleg_outputs}
ent_emb_file=${2:-None}
static_ent_emb_file=${3:-None}
log_name=${4:-run}
bert_model=${5:-spanbert-large-cased}
batch_size=${6:-16}
valid_batch_size=${7:-256}
epoch=${8:-10}
lr=${9:-0.00002}
l2=${10:-0.01}
warmup_percentage=${11:-0.1}
gradient_accumulation_steps=${12:-4}
max_seq_length=${13:-128}
seed=${14:-1234}

python run.py \
     --seed ${seed} \
     --data_dir ${data_dir} \
     --ent_emb_file ${ent_emb_file} \
     --static_ent_emb_file ${static_ent_emb_file} \
     --log_path logs/${bert_model}_${log_name} \
     --bert_model ${bert_model} \
     --n_epochs ${epoch} \
     --batch_size ${batch_size} \
     --valid_batch_size ${valid_batch_size} \
     --gradient_accumulation_steps ${gradient_accumulation_steps} \
     --optimizer bert_adam \
     --bert_adam_eps 1e-6 \
     --lr ${lr} \
     --l2 ${l2} \
     --warmup_percentage ${warmup_percentage} \
     --valid_split dev test \
     --checkpointing 1 \
     --checkpoint_metric TACRED/TACRED/dev/F1:max \
     --checkpoint_task_metrics TACRED/TACRED/test/F1:max \
     --counter_unit batch \
     --max_seq_length ${max_seq_length} \
     --evaluation_freq 850