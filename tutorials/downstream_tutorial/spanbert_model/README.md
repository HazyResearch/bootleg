# Bootleg + SpanBERT

This repository shows how to integrate Bootleg embeddings in a SpanBERT model and is built using the [Emmental framework](https://emmental.readthedocs.io/en/latest/user/getting_started.html), a pytorch wrapper which defines a consistent code structure across Emmental applications. 

[Prior work](https://arxiv.org/abs/1907.10529) uses SpanBERT to achieve strong results on TACRED, so we use this as our foundation.

## Execution 
We provided a run.sh script, though the commands for launching training are provided again here: 
```
python run.py \
    --seed 1234 \
    --data_dir datasets/ \
    --ent_emb_file datasets/ent_embedding.npy \
    --static_ent_emb_file None \
    --log_path logs/spanbert-large-cased \
    --bert_model spanbert-large-cased \
    --n_epochs 10 \
    --batch_size 8 \
    --valid_batch_size 200 \
    --gradient_accumulation_steps 4 \
    --optimizer bert_adam \
    --bert_adam_eps 1e-6 \
    --lr 0.00002 \
    --l2 0.01 \
    --warmup_percentage 0.1 \
    --valid_split dev test \
    --checkpointing 1 \
    --checkpoint_metric TACRED/TACRED/dev/F1:max \
    --checkpoint_task_metrics TACRED/TACRED/test/F1:max \
    --counter_unit batch \
    --max_seq_length 128 \
    --evaluation_freq 425 \
    --encode_first 0 \
    --tanh 0 \
    --norm 0 &
```

If training on a machine with multiple GPUs, use `CUDA_VISIBLE_DEVICES`.

## Details

We encode the input example via SpanBERT-large, then concat the result with the Bootleg embeddings. We run this through four transformer layers (see `encoder.py`) before the classification and pooling steps. To view the full task flow, see `task.py`. We found that combining the Bootleg embeddings directly in with the word embeddings led to worse performance than including the Bootleg embeddings after forming the word embeddings. 