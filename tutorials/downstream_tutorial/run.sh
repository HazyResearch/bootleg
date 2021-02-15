data_dir=${1:-dataset/tacred}
bert_model=${2:-spanbert-large-cased}
batch_size=${3:-8}
valid_batch_size=${4:-200}
epoch=${5:-10}
lr=${6:-0.00002}
l2=${7:-0.01}
warmup_percentage=${8:-0.1}
gradient_accumulation_steps=${9:-4}
max_seq_length=${10:-128}
seed=${11:-1234}
encode_first=${12:-0}
tanh=${13:-0}
norm=${14:-0}
ent_emb_path=${15:-dataset/tacred/ent_embedding.npy} 

#!/bin/bash
echo -n "Model [spanbert, lstm]: "
read VAR

if [[ "$VAR" = "spanbert" ]]; then
    echo "You chose spanbert"
    python spanbert_model/run.py \
         --seed ${seed} \
         --data_dir ${data_dir} \
         --ent_emb_file ${ent_emb_path} \
         --static_ent_emb_file None \
         --log_path logs/${bert_model} \
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
         --counter_unit batch \
         --max_seq_length ${max_seq_length} \
         --evaluation_freq 425 \
         --encode_first ${encode_first} \
         --tanh ${tanh} \
         --norm ${norm}
elif [[ "$VAR" = "lstm" ]]; then
  echo "lstm"
  python lstm_model/train.py \
      --use_ctx_ent 
else
  echo $VAR
fi
