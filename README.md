# eunn_seq2seq
This is the implementation of EUNN for seq2seq learning. [EUNN paper](https://arxiv.org/pdf/1612.05231.pdf). 

## Usage
Generate toy reverse data for seq2seq training
```
gen_data.py
```
Run models
```
# for normal lstm seq2seq model
python3 seq2seq.py
# for complex/orthogonal seq2seq model
python3 complex.py
```
Tensorboard for visualization
```
tensorboard --logdir {desired repository}
```

EUNN cell codes borrow from [EUNN-tensorflow](https://github.com/jingli9111/EUNN-tensorflow/edit/master/README.md)
