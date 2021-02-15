DATA_DIR=$1

if [ $# -eq 0 ]
  then
    echo "Need to include data directory as 'bash download_wiki.sh <data_dir>'"
	exit 1
fi

mkdir -p $DATA_DIR

wget https://bootleg-data.s3.amazonaws.com/data/latest/wiki.tar.gz -P $DATA_DIR

tar -xzvf $DATA_DIR/wiki.tar.gz -C $DATA_DIR
