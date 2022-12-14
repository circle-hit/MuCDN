# MuCDN

* Code for [COLING 2022](https://coling2022.org) accepted paper titled "MuCDN: Mutual Conversational Detachment Network for Emotion Recognition in Multi-Party Conversations"
* Weixiang Zhao, Yanyan Zhao, Bing Qin.

## Requirements
* Python 3.7
* PyTorch 1.8.2
* Transformers 4.12.3
* CUDA 11.1

## Preparation

## Discourse Parsing 
We use a powerful dialogue discourse parser to obtain the structure for the explicit detachment.

Details for training the parser could be found in https://github.com/shizhouxing/DialogueDiscourseParsing. We have already processed the ERC data in the same formmat with the input of the paser, which is in the folder of erc_data.

With the trained parser, run main_inference.py to get the parsed ERC data. 

You can also direcly use the preprocessed features in the following.

### Preprocessed Features
You can download the preprocessed features including dataset, extracted utterance feature and dialogue discourse structure we used from:
https://pan.baidu.com/s/1gMIyK4mXVSvis1f1_DSQsQ  提取码:j84f

or from: https://drive.google.com/drive/folders/1-VzZjNmdZMO9rzzQD-JOt46w53fsBwJc?usp=sharing

and place them into the corresponding folds like emorynlp and meld

## Training
You can train the models with the following codes:

For EmoryNLP: 
`python train_emorynlp.py --hidden_dim 300 --pos`

For MELD: 
`python train_meld.py --hidden_dim 300 --pos --norm`
