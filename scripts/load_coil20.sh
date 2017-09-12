DATASET_NAME=coil-20-proc
DATA_URL=http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip
mkdir -p data
wget -p data $DATA_URL
unzip data/$DATASET_NAME -d data
rm data/$DATASET_NAME.zip
