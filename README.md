# NLP-BERT 谷歌自然语言处理模型：BERT-基于pytorch
**Notice: This is only For the convinience of Chineses reader who cannot read English version directly**

## 完整中文版BERT模型分析请移步我的博客：[NLP自然语言处理-谷歌BERT模型深度解析](https://blog.csdn.net/qq_39521554/article/details/83062188)
Author-作者
Junseong Kim, Scatter Lab (codertimo@gmail.com / junseong.kim@scatter.co.kr)
License
This project following Apache 2.0 License as written in LICENSE file

Copyright 2018 Junseong Kim, Scatter Lab, respective BERT contributors  
Copyright (c) 2018 Alexander Rush : The Annotated Trasnformer

 Environment require:  
               -tqdm  
               -numpy  
               -torch>=0.4.0  
               -python3.6+  

This version has based on the version in https://github.com/codertimo/BERT-pytorch  
此中文版本仅基于原作者Junseong Kim与原Google项目的pytorch版本代码作为分享，如有其他用途请与原作者联系  

[![LICENSE](https://img.shields.io/github/license/codertimo/BERT-pytorch.svg)](https://github.com/codertimo/BERT-pytorch/blob/master/LICENSE)
![GitHub issues](https://img.shields.io/github/issues/codertimo/BERT-pytorch.svg)
[![GitHub stars](https://img.shields.io/github/stars/codertimo/BERT-pytorch.svg)](https://github.com/codertimo/BERT-pytorch/stargazers)
[![CircleCI](https://circleci.com/gh/codertimo/BERT-pytorch.svg?style=shield)](https://circleci.com/gh/codertimo/BERT-pytorch)
[![PyPI](https://img.shields.io/pypi/v/bert-pytorch.svg)](https://pypi.org/project/bert_pytorch/)
[![PyPI - Status](https://img.shields.io/pypi/status/bert-pytorch.svg)](https://pypi.org/project/bert_pytorch/)
[![Documentation Status](https://readthedocs.org/projects/bert-pytorch/badge/?version=latest)](https://bert-pytorch.readthedocs.io/en/latest/?badge=latest)

Pytorch implementation of Google AI's 2018 BERT, with simple annotation

> BERT 2018 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
> Paper URL : https://arxiv.org/abs/1810.04805


## 介绍

最近谷歌搞了个大新闻，公司AI团队新发布的BERT模型，在机器阅读理解顶级水平测试SQuAD1.1中表现出惊人的成绩：全部两个衡量指标上全面超越人类，并且还在11种不同NLP测试中创出最佳成绩，包括将GLUE基准推至80.4％（绝对改进7.6％），MultiNLI准确度达到86.7% （绝对改进率5.6％）等。可以预见的是，BERT将为NLP带来里程碑式的改变，也是NLP领域近期最重要的进展。

# BERT模型具有以下两个特点：

第一，是这个模型非常的深，12层，并不宽(wide），中间层只有1024，而之前的Transformer模型中间层有2048。这似乎又印证了计算机图像处理的一个观点——深而窄 比 浅而宽 的模型更好。

第二，MLM（Masked Language Model），同时利用左侧和右侧的词语，这个在ELMo上已经出现了，绝对不是原创。其次，对于Mask（遮挡）在语言模型上的应用，已经被Ziang Xie提出了（我很有幸的也参与到了这篇论文中）：[1703.02573] Data Noising as Smoothing in Neural Network Language Models。这也是篇巨星云集的论文：Sida Wang，Jiwei Li（香侬科技的创始人兼CEO兼史上发文最多的NLP学者），Andrew Ng，Dan Jurafsky都是Coauthor。但很可惜的是他们没有关注到这篇论文。用这篇论文的方法去做Masking，相信BRET的能力说不定还会有提升。


部分代码基于 [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
目前此项目仍在开发中，后续会有不断更新，欢迎ＦＯＲＫ

## 安装
```
pip install bert-pytorch
```

## 快速启动

**注意 : 你的语料如果位于同一行，则应该由制表符\t所分割**

### 0. 准备你的语料（此处为英文语料）
```
Welcome to the \t the jungle\n
I can stay \t here all night\n
```

or tokenized corpus (tokenization is not in package)
```
Wel_ _come _to _the \t _the _jungle\n
_I _can _stay \t _here _all _night\n
```


### 1. 建立词汇
```shell
bert-vocab -c data/corpus.small -o data/vocab.small
```

### 2. 训练BERT模型
```shell
bert -c data/corpus.small -v data/vocab.small -o output/bert.model
```

## 语言模型预训练

在原论文中，作者展示了新的语言训练模型，称为编码语言模型与次一句预测
In the paper, authors shows the new language model training methods, 
which are "masked language model" and "predict next sentence".


### 编码语言模型

> Original Paper : 3.3.1 Task #1: Masked LM 

```
Input Sequence  : The man went to [MASK] store with [MASK] dog
Target Sequence :                  the                his
```

#### 规则:
会有15%的随机输入被改变，这些改变基于以下规则：
Randomly 15% of input token will be changed into something, based on under sub-rules：

1. 80%的tokens会成为‘掩码’token
2. 10%的tokens会称为‘随机’token
3. 10%的tokens会保持不变但需要被预测


### 次一句预测

> Original Paper : 3.3.2 Task #2: Next Sentence Prediction

```
Input : [CLS] the man went to the store [SEP] he bought a gallon of milk [SEP]
Label : Is Next

Input = [CLS] the man heading to the store [SEP] penguin [MASK] are flight ##less birds [SEP]
Label = NotNext
```

"Is this sentence can be continuously connected?"


#### 规则:

1. 50%的下一句会（随机）成为连续句子
2. 50%的下一句会（随机）成为不关联句子


## 作者
Junseong Kim, Scatter Lab (codertimo@gmail.com / junseong.kim@scatter.co.kr)

## License

This project following Apache 2.0 License as written in LICENSE file

Copyright 2018 Junseong Kim, Scatter Lab, respective BERT contributors

Copyright (c) 2018 Alexander Rush : [The Annotated Trasnformer](https://github.com/harvardnlp/annotated-transformer)
