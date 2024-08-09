# Dependencies
python==3.10.14
torch==2.2.2+cu121
torch-geometric== 2.5.3

## Dataset
You can find the raw data at [IEMOCAP](https://sail.usc.edu/iemocap/) and [MELD](https://github.com/declare-lab/MELD)

## Preparing datasets for training

    python preprocess.py --data './data/newdata.pkl' --dataset="iemocap"

## Training networks 

    python train.py --data=./data/newdata.pkl --from_begin --device=cuda:0 --epochs=150 --batch_size=20

## Predictioning networks 

    python prediction.py --data=./data/newdata.pkl --device=cuda --epochs=1 --batch_size=20


## Performance Comparison IEMOCAP

|Dataset | Weighted F1(w) | Acc.   |
|--------|----------------|--------|
|IEMOCAP | 71.04%         | 70.98% |

## Performance Comparison MELD

|Dataset | Weighted F1(w) | Acc.   |
|--------|----------------|--------|
|MELD    | 65.40%         | 66.53% |

