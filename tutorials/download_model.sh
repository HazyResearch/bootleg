MODEL_DIR=models
MODEL=${1:-uncased}

mkdir -p $MODEL_DIR

wget https://bootleg-data.s3.amazonaws.com/models/latest/bootleg_$MODEL.tar.gz -P $MODEL_DIR

tar -xzvf $MODEL_DIR/bootleg_$MODEL.tar.gz -C $MODEL_DIR
