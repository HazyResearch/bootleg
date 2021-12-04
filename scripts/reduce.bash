source ../venv/bin/activate
source ./envs.bash
rm -rf $BOOTLEG_DATA_DIR/output/full_wiki_0_-1_final/_temp
python3 $BOOTLEG_CODE_DIR/bootleg/utils/preprocessing/sample_eval_data.py \
    --data_dir $BOOTLEG_DATA_DIR/output/full_wiki_0_-1_final \
    --file dev.jsonl \
    --out_file_name dev_test_sample.jsonl \
    --sample_perc 0.1 \
    --min_sample_size 5000 \
    --slice unif_NS_all \
    --slice unif_NS_TO \
    --slice unif_NS_TS \
    --slice unif_NS_TL \
