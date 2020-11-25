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
1. Training instructions will be uploaded here soon!