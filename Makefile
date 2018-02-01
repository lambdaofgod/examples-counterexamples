prepare_coil20:
	bash load_coil20.sh

# download word embedding data (with vectors from GloVe)
load_word_embeddings:
	bash scripts/load_word_embeddings.sh 

# convert GloVe word embeddings to word2vec format
# example: prepare_embeddings in_file_name=glove.6B.100d.txt out_file_name=glove.6B.100d.w2v
prepare_embeddings:
	python src/convert_glove_2_word2vec.py $(in_file_name) $(out_file_name) 

load_101_categories:
	bash scripts/load_101_categories.sh
