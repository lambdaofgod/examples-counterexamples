from os.path import join
from sys import argv

import glove_2_word2vec
import load_data_utils


__, DATA_DIR = load_data_utils.get_env_vars()
VECTORS_DIR = join(DATA_DIR, 'glove.6B')
in_file_name = argv[1]
out_file_name = argv[2]


if __name__ == '__main__':
    input_path = join(VECTORS_DIR, in_file_name)
    model = glove_2_word2vec.load_word2vec_model(input_path)
    print()
    print('loaded glove model from')
    print(str(input_path))
    
    output_path = join(VECTORS_DIR, out_file_name)
    model.save(output_path)
    print()
    print('saved glove model to')
    print(str(out_file_name))
