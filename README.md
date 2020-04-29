# Recent Advances in Vision and Language PreTrained Models (VL-PTMs)
Maintained by [WANG Yue](https://yuewang-cuhk.github.io/) (yuewang@cse.cuhk.edu.hk). Last update on 2020/04/29.

## Table of Contents

* [Image-based VL-PTMs](#image-based-vl-ptms)
  * [Representation Learning](#representation-learning)
  * [Task-specific](#task-specific)
  * [Other Analysis](#other-analysis)
* [Video-based VL-PTMs](#video-based-vl-ptms)
* [Speech-based VL-PTMs](#speech-based-vl-ptms)
* [Other Resources](#other-resources)


# Image-based VL-PTMs

## Representation Learning

[ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks](https://arxiv.org/abs/1908.02265), NeurIPS 2019 [[code]](https://github.com/jiasenlu/vilbert_beta)

[LXMERT: Learning Cross-Modality Encoder Representations from Transformers](https://arxiv.org/abs/1908.07490), EMNLP 2019 [[code]](https://github.com/airsplay/lxmert)

[VL-BERT: Pre-training of Generic Visual-Linguistic Representations](https://arxiv.org/abs/1908.08530), ICLR 2020
 [[code]](https://github.com/jackroos/VL-BERT)

[VisualBERT: A Simple and Performant Baseline for Vision and Language](https://arxiv.org/abs/1908.03557), arXiv 2019/08 [[code]](https://github.com/uclanlp/visualbert)

[Unicoder-VL: A Universal Encoder for Vision and Language by Cross-modal Pre-training](https://arxiv.org/abs/1908.06066), AAAI 2020

[Unified Vision-Language Pre-Training for Image Captioning and VQA](https://arxiv.org/pdf/1909.11059.pdf), AAAI 2020, [[code]](https://github.com/LuoweiZhou/VLP), (**VLP**)

[UNITER: Learning Universal Image-text Representations](https://arxiv.org/abs/1909.11740), arXiv 2019/09 [[code]](https://github.com/ChenRocks/UNITER)

[Weak Supervision helps Emergence of Word-Object Alignment and improves Vision-Language Tasks](https://arxiv.org/abs/1912.03063), arXiv 2019/12

[InterBERT: Vision-and-Language Interaction for Multi-modal Pretraining](https://arxiv.org/abs/2003.13198), arXiv 2020/03

[Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks](https://arxiv.org/pdf/2004.06165.pdf), arXiv 2020/04

[Pixel-BERT: Aligning Image Pixels with Text by Deep Multi-Modal Transformers](https://arxiv.org/abs/2004.00849), arXiv 2020/04


## Task-specific

**VCR**: [Fusion of Detected Objects in Text for Visual Question Answering](https://arxiv.org/abs/1908.05054), EMNLP 2019, [[code]](https://github.com/google-research/language/tree/master/language/question_answering/b2t2), (**B2T2**)

**TextVQA**: [Iterative Answer Prediction with Pointer-Augmented Multimodal Transformers for TextVQA](https://arxiv.org/abs/1911.06258), CVPR 2020, [[code]](https://github.com/ronghanghu/pythia/tree/project/m4c/projects/M4C), (**M4C**)

**VisDial**: [Large-scale Pretraining for Visual Dialog: A Simple State-of-the-Art Baseline](https://arxiv.org/abs/1912.02379), arXiv 2019/12, [[code]](https://github.com/vmurahari3/visdial-bert), (**VisDial-BERT**)

**VisDial**: [VD-BERT: A Unified Vision and Dialog Transformer with BERT](https://arxiv.org/abs/2004.13278), arXiv 2020/04 (**SOTA** in the VisDial leaderboard)

**VLN**: [Towards Learning a Generic Agent for Vision-and-Language Navigation via Pre-training](https://arxiv.org/abs/2002.10638), CVPR 2020, [[code]](https://github.com/weituo12321/PREVALENT), (**PREVALENT**)

**Text-image retrieval**: [ImageBERT: Cross-Modal Pre-training with
Large-scale Weak-supervised Image-text Data](https://arxiv.org/abs/2001.07966), arXiv 2020/01

**Image captioning**: [XGPT: Cross-modal Generative Pre-Training for Image Captioning](https://arxiv.org/abs/2003.01473), arXiv 2020/03

**Visual Question Generation**: [BERT Can See Out of the Box: On the Cross-modal Transferability of Text Representations](https://arxiv.org/abs/2002.10832), arXiv 2020/02


## Other Analysis

**Multi-task Learning**, [12-in-1: Multi-Task Vision and Language Representation Learning](https://arxiv.org/abs/1912.02315), CVPR 2020, [[code]](https://github.com/facebookresearch/vilbert-multi-task) 

**Social Bias in VL Embedding**, [Measuring Social Biases in Grounded Vision and Language Embeddings](https://arxiv.org/abs/2002.08911), arXiv 2020/02, [[code]](https://github.com/candacelax/bias-in-vision-and-language)
  
  
# Video-based VL-PTMs

[VideoBERT: A Joint Model for Video and Language Representation Learning](https://arxiv.org/abs/1904.01766), ICCV 2019

[Learning Video Representations Using Contrastive Bidirectional Transformers](https://arxiv.org/abs/1906.05743), arXiv 2019/06, (**CBT**)

[M-BERT: Injecting Multimodal Information in the BERT Structure](https://arxiv.org/abs/1908.05787), arXiv 2019/08

[BERT for Large-scale Video Segment Classification with Test-time Augmentation](https://arxiv.org/abs/1912.01127), 	ICCV 2019 YouTube8M workshop, [[code]](https://github.com/hughshaoqz/3rd-Youtube8M-TM)

[Bridging Text and Video: A Universal Multimodal Transformer for Video-Audio Scene-Aware Dialog](https://arxiv.org/abs/2002.00163), AAAI2020 DSTC8 workshop

[UniViLM: A Unified Video and Language Pre-Training Model for Multimodal Understanding and Generation](https://arxiv.org/abs/2002.06353), arXiv 2020/02


# Speech-based VL-PTMs

[Towards Transfer Learning for End-to-End Speech Synthesis from Deep Pre-Trained Language Models](https://arxiv.org/abs/1906.07307), arXiv 2019/06

[Understanding Semantics from Speech Through Pre-training](https://arxiv.org/abs/1909.10924), arXiv 2019/09

[SpeechBERT: Cross-Modal Pre-trained Language Model for End-to-end Spoken Question Answering](https://arxiv.org/abs/1910.11559), arXiv 2019/10

[vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations](https://arxiv.org/abs/1910.05453),  arXiv 2019/10

[Effectiveness of self-supervised pre-training for speech recognition](https://arxiv.org/abs/1911.03912),  arXiv 2019/11


# Other Resources

* Two recent surveys on pretrained language models
  * [Pre-trained Models for Natural Language Processing: A Survey](https://arxiv.org/abs/2003.08271), arXiv 2020/03
  * [A Survey on Contextual Embeddings](https://arxiv.org/abs/2003.07278), arXiv 2020/03
* Other surveys about multimodal research
  * [Trends in Integration of Vision and Language Research: A Survey of Tasks, Datasets, and Methods](https://arxiv.org/abs/1907.09358), arXiv 2019
  * [Deep Multimodal Representation Learning: A Survey](https://ieeexplore.ieee.org/abstract/document/8715409), arXiv 2019 
  * [Multimodal Machine Learning: A Survey and Taxonomy](https://arxiv.org/abs/1705.09406), TPAMI 2018
  * [A Comprehensive Survey of Deep Learning for Image Captioning](https://arxiv.org/abs/1810.04020), ACM Computing Surveys 2018
* Other repositories of relevant reading list
  * [Pre-trained Languge Model Papers from THU-NLP](https://github.com/thunlp/PLMpapers)
  * [BERT-related Papers](https://github.com/tomohideshibata/BERT-related-papers)
  * [Reading List for Topics in Multimodal Machine Learning](https://github.com/pliang279/awesome-multimodal-ml)
  * [A repository of vision and language papers](https://github.com/sangminwoo/awesome-vision-and-language-papers)

