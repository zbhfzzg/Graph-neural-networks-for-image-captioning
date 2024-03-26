# Graph-neural-networks-for-image-captioning
Final Year Prject 
##Version 1

激活Anaconda prompt中的环境，需要先管理员启动
conda activate pytorch_dl
缺乏package的话需要通过Anaconda prompt来安装。别的没有用的。

教程1
https://www.datacamp.com/tutorial/comprehensive-introduction-graph-neural-networks-gnns-tutorial

教程2 keras的
https://www.analyticsvidhya.com/blog/2021/12/step-by-step-guide-to-build-image-caption-generator-using-deep-learning/

教程3
https://ar5iv.labs.arxiv.org/html/1707.07998

内网穿透：tailscale
ssh remote command

3.20 
学习project： https://github.com/RoyalSkye/Image-Caption
Tut： https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

Attention. The use of Attention networks is widespread in deep learning, and with good reason. This is a way for a model to choose only those parts of the encoding that it thinks is relevant to the task at hand. The same mechanism you see employed here can be used in any model where the Encoder's output has multiple points in space or time. In image captioning, you consider some pixels more important than others. In sequence to sequence tasks like machine translation, you consider some words more important than others.


## Data set
MSCOCO   https://pjreddie.com/projects/coco-mirror/#google_vignette

Fliker 30K dataset. Because MSCOCO is too large
Images and captions link     https://www.kaggle.com/datasets/adityajn105/flickr30k

Fliker 8K    https://www.kaggle.com/datasets/adityajn105/flickr8k


## py文件用途
utils.py: 这个文件很可能包含各种工具函数，比如图片预处理、数据加载辅助函数或其他通用功能。这些工具函数可以被项目中的其他Python文件调用。

caption.py: 根据文件名，这个文件可能用于处理或生成图片的描述（caption）。它可能包含模型的部分逻辑，专门用于处理文本生成或文本处理的任务。

create_input_files.py: 这个文件可能用于数据预处理阶段，特别是将原始数据转换成模型训练、验证或测试所需的格式。这可能包括图片的预处理和与之相关的文本描述。

datasets.py: 通常，这个文件会定义一个或多个PyTorch Dataset类，用于封装数据集的加载逻辑，使其可以被DataLoader使用，以便在训练和评估模型时批量加载数据。

eval.py: 这个文件很可能包含模型评估的逻辑，如计算和报告模型性能的各种指标。

models.py: 该文件应定义模型的架构，包括使用的神经网络层和前向传播逻辑。基于您的描述，可能包括预训练的ImageNet模型作为编码器的一部分。

train.py: 这个文件应包含模型的训练逻辑，包括训练循环、损失函数的计算、模型参数的更新等。

Json文件也就是dataset_flickr8k.json   token是分词，raw是原始字幕。


## 运行步骤：
you can looke for the steps with carefully reading in the readme
1、dowonload the datastes and modify the paths in the create_input_files.py
2、run the create_input_file.py    这一步让训练集、验证集和测试集的图像及其字幕都已经被读取并且存储到了文件中
3、run the train.py
4、load the model to test the image of yourself
there are some wrong in the processing,you can find the anwser in the issues


Train.py  基于注意力机制的图像标题生成模型的训练和验证过程

## Model.py
这个 model.py 文件包含了三个主要的神经网络模块，分别用于在图像描述任务中编码图像、注意力机制和解码文本描述。下面是每个模块的简要说明和如何在其中加入图形神经网络（GNN）的建议。

Encoder
编码器是基于ResNet-101模型，用于从输入图像中提取特征。这个过程包括通过ResNet的卷积层传递图像，并使用自适应平均池化层将输出调整为固定大小。编码器的输出是一个维度为 (batch_size, encoded_image_size, encoded_image_size, 2048) 的张量，表示图像的编码特征。

Attention
注意力模块采用编码器的输出和上一个时间步的解码器隐藏状态作为输入，计算当前时间步对于图像中不同区域的注意力权重。这有助于解码器集中于图像的特定部分来生成下一个单词。

DecoderWithAttention
解码器模块使用LSTM网络和注意力机制来生成图像的文本描述。它首先使用嵌入层将输入的文本标记转换为向量，然后结合编码器输出的图像特征和先前的隐藏状态来生成下一个单词的预测。