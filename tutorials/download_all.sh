DATA_DIR=$1

if [ $# -eq 0 ]
  then
    echo "Need to include data directory as 'bash download_all.sh <data_dir>'"
	exit 1
fi

mkdir -p $DATA_DIR

wget https://bootleg-emb.s3.amazonaws.com/models/2020_08_25/bootleg_wiki.tar.gz -P $DATA_DIR
wget https://bootleg-emb.s3.amazonaws.com/emb_data.tar.gz -P $DATA_DIR
wget https://bootleg-emb.s3.amazonaws.com/data/rss500.tar.gz -P $DATA_DIR
wget https://bootleg-emb.s3.amazonaws.com/data/nq.tar.gz -P $DATA_DIR
wget https://bootleg-emb.s3.amazonaws.com/entity_db.tar.gz -P $DATA_DIR

tar -xzvf $DATA_DIR/bootleg_wiki.tar.gz -C $DATA_DIR
tar -xzvf $DATA_DIR/emb_data.tar.gz -C $DATA_DIR
tar -xzvf $DATA_DIR/rss500.tar.gz -C $DATA_DIR
tar -xzvf $DATA_DIR/nq.tar.gz -C $DATA_DIR
tar -xzvf $DATA_DIR/entity_db.tar.gz -C $DATA_DIR
