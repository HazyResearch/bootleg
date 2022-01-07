source ./envs.bash

BOOTLEG_TRAIN_DATA_DIR=$BOOTLEG_BASE_DIR/output/kore50_aliases_filtered_0_-1_final
BOOTLEG_ENTITYDB_DIR=$BOOTLEG_TRAIN_DATA_DIR/entity_db
KORE50_CONFIG=$BOOTLEG_CONFIGS_DIR/base_config.yaml

cp $KORE50_CONFIG /tmp/base_config.yaml
sed -i "s|BOOTLEG_LOGS_DIR|$BOOTLEG_LOGS_DIR|g" /tmp/base_config.yaml
sed -i "s|BOOTLEG_TRAIN_DATA_DIR|$BOOTLEG_TRAIN_DATA_DIR|g" /tmp/base_config.yaml
sed -i "s|BOOTLEG_BERT_CACHE_DIR|$BOOTLEG_BERT_CACHE_DIR|g" /tmp/base_config.yaml
sed -i "s|BOOTLEG_ENTITYDB_DIR|$BOOTLEG_ENTITYDB_DIR|g" /tmp/base_config.yaml

cat /tmp/base_config.yaml
echo ''
echo 'Starting to train...'
echo ''
cd $BOOTLEG_CODE_DIR
python3 ./bootleg/run.py --config_script /tmp/base_config.yaml
