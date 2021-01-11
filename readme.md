# Grounded Latent Trees for Visual Question Answering 

This repository contains the code associated with the paper [Latent Compositional Representations Improve Systematic Generalization in Grounded Question Answering](https://arxiv.org/abs/2007.00266), published in TACL (2020).

## Setup
1. Install all requirements
```
pip install -r requirements.txt
```

## Arithmetic Dataset
2. Run with this AllenNLP command:
```
allennlp train train_configs/glt_arithmetic.jsonnet -s experiments/experiment_name --include-package src
```
## CLEVR / CLOSURE Dataset
1. Download [CLEVR image features + preprocessed data](https://drive.google.com/file/d/1TQSfMiiNuGza1muu09py03P_DP1r64QI/view?usp=sharing) (image features are extracted with [py-bottom-up-attention](https://github.com/airsplay/py-bottom-up-attention)) into the `data` directory.

2. Run with this AllenNLP command:
```
allennlp train train_configs/glt_clevr.jsonnet -s experiments/experiment_name --include-package src
```