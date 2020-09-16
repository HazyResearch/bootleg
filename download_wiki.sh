DATA_DIR=$1

if [ $# -eq 0 ]
  then
    echo "Need to include data directory as 'bash download_all.sh <data_dir>'"
	exit 1
fi

mkdir -p $DATA_DIR

wget https://bootleg-emb.s3.amazonaws.com/data/wiki.tar.gz -P $DATA_DIR

tar -xzvf $DATA_DIR/wiki.tar.gz -C $DATA_DIR
