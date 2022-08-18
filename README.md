# MuCDN
## Requirements
* Python 3.7
* PyTorch 1.8.2
* Transformers 4.12.3
* CUDA 11.1

## Preparation

### Preprocessed Features
You can download the preprocessed features including dataset, extracted utterance feature and dialogue discourse structure we used from:
https://pan.baidu.com/s/1gMIyK4mXVSvis1f1_DSQsQ  提取码:j84f

and place them into the corresponding folds like emorynlp and meld

## Training
You can train the models with the following codes:

For EmoryNLP: 
`python train_emorynlp.py --hidden_dim 300 --pos`

For MELD: 
`python train_meld.py --hidden_dim 300 --pos --norm`