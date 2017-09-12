DATASET_NAME=glove.6B
WORD_VECTORS_URL=http://nlp.stanford.edu/data/$DATASET_NAME.zip

mkdir -p data
wget -P data $WORD_VECTORS_URL
unzip data/$DATASET_NAME.zip -d data/$DATASET_NAME
rm data/$DATASET_NAME.zip

python src/convert_glove_2_word2vec.py glove.6B.100d.txt word2vec.6B.100d

