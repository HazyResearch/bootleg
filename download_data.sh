DATA_DIR=data

mkdir -p $DATA_DIR

wget https://bootleg-emb.s3.amazonaws.com/data/emb_data.tar.gz -P $DATA_DIR
wget https://bootleg-emb.s3.amazonaws.com/data/rss500.tar.gz -P $DATA_DIR
wget https://bootleg-emb.s3.amazonaws.com/data/nq.tar.gz -P $DATA_DIR
wget https://bootleg-emb.s3.amazonaws.com/data/wiki_entity_data.tar.gz -P $DATA_DIR

tar -xzvf $DATA_DIR/emb_data.tar.gz -C $DATA_DIR
tar -xzvf $DATA_DIR/rss500.tar.gz -C $DATA_DIR
tar -xzvf $DATA_DIR/nq.tar.gz -C $DATA_DIR
tar -xzvf $DATA_DIR/wiki_entity_data.tar.gz -C $DATA_DIR
