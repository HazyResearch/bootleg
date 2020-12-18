MODEL_DIR=models

mkdir -p $MODEL_DIR

wget https://bootleg-emb.s3.amazonaws.com/models/2020_12_09/bootleg_wiki.tar.gz -P $MODEL_DIR

tar -xzvf $MODEL_DIR/bootleg_wiki.tar.gz -C $MODEL_DIR
