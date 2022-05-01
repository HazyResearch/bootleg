DATA_DIR=data

mkdir -p $DATA_DIR

wget https://bootleg-ned-data.s3-us-west-1.amazonaws.com/data/latest/nq.tar.gz -P $DATA_DIR
wget https://bootleg-ned-data.s3-us-west-1.amazonaws.com/data/latest/entity_db.tar.gz -P $DATA_DIR

tar -xzvf $DATA_DIR/nq.tar.gz -C $DATA_DIR
tar -xzvf $DATA_DIR/entity_db.tar.gz -C $DATA_DIR
