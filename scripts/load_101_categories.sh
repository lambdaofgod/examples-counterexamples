# get 101 categories data
wget -P data http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
tar -xvzf data/101_ObjectCategories.tar.gz -C data
rm data/101_ObjectCategories.tar.gz
