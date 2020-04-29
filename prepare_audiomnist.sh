echo "Downloading AudioMNIST..."
python -c "from art.utils import get_file; get_file('audiomnist.tar.gz', 'https://api.github.com/repos/soerenab/AudioMNIST/tarball')"

echo "Untar..."
export DATASET_TMP=/tmp/.art/audiomnist.tar.gz
tar zxf $DATASET_TMP

echo "Split into Training/Test set..."
export DATASET_DIR=data/audiomnist
mkdir -p $DATASET_DIR/train $DATASET_DIR/test
for i in {01..47}; do
    mv soerenab-AudioMNIST-3c9ed8c/data/$i $DATASET_DIR/train
done

for i in {48..60}; do
    mv soerenab-AudioMNIST-3c9ed8c/data/$i $DATASET_DIR/test
done

echo "Cleaning up"
rm -r soerenab-AudioMNIST-3c9ed8c
# rm $DATASET_TMP
