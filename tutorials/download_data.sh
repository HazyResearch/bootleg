DATA_DIR=data

mkdir -p $DATA_DIR

wget https://bootleg-data.s3.amazonaws.com/data/latest/emb_data.tar.gz -P $DATA_DIR
wget https://bootleg-data.s3.amazonaws.com/data/latest/nq.tar.gz -P $DATA_DIR
wget https://bootleg-data.s3.amazonaws.com/data/latest/wiki_entity_data.tar.gz -P $DATA_DIR

tar -xzvf $DATA_DIR/emb_data.tar.gz -C $DATA_DIR
tar -xzvf $DATA_DIR/nq.tar.gz -C $DATA_DIR
tar -xzvf $DATA_DIR/wiki_entity_data.tar.gz -C $DATA_DIR
