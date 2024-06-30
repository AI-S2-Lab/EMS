# Emotion-Aware Speech Self-Supervised Representation Learning with Intensity Knowledge
 

## Introduction
This is an implementation of the following paper.
> [Emotion-Aware Speech Self-Supervised Representation Learning with Intensity Knowledge.](https://www.arxiv.org/abs/2406.06646)
 

 [Rui Liu](https://ttslr.github.io/), [Zening Ma](https://github.com/codening99).
 

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)]([https://www.arxiv.org/abs/2406.06646])


 
## Environment
install s3prl package:
```
pip install s3prl
```
Our code is modified from S3PRL, more details of S3PRL installation can be found at https://s3prl.github.io/s3prl/tutorial/installation.html#


## Usage
You can test for downstream tasks using the following code:
```
python run_downstream.py -m train -d downstream_name -n your_name -u upstream_name -k xxx.ckpt
```
`-u` specifies the upstream pretrained model. In our approach you can use `npc_local` or `mockingjay_local`.

`-d` specifies the downstream task. There are three downstream tasks included in our approach, which are: `emotion`, `fluent_commands`, `voxceleb1` and `ctc`.


More details on downstream missions can be found at: https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/README.md
