load_word_embeddings:
	bash scripts/load_word_embeddings.sh 

prepare_embeddings:
	python src/convert_glove_2_word2vec.py $(in_file_name) $(out_file_name) 


