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

修改：
# 假设已经有了 GCN 模块的实现
class GCN(nn.Module):
    def __init__(self, in_features, out_features, num_pixels):
        super(GCN, self).__init__()
        # 这里添加你的 GCN 层初始化代码
        self.gcn_layer = ...  # 你的 Graph Convolutional Layer

    def forward(self, x):
        # 假设 x 的形状是 (batch_size, in_features, num_pixels, num_pixels)
        # 在这里应用 GCN 操作
        gcn_output = self.gcn_layer(x)
        # 确保 GCN 输出与输入有相同的形状
        return gcn_output

# Based on restnet 101 model with GCN
class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14): #encoded_image_size=14 feature map size  
        num_pixels = encoded_image_size * encoded_image_size
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        #resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101 已经不能够使用了，需要改为weight
        resnet = torchvision.models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

         # 假设你的GCN层将14x14x2048的特征图转换为同样大小的输出
        self.gcn = GCN(in_features=2048, out_features=2048, num_pixels=num_pixels)


        # 你可能需要初始化一个邻接矩阵来表示图的结构
        self.adj_matrix = self.create_adj_matrix(encoded_image_size * encoded_image_size)

        self.fine_tune()

    def forward(self, images):   #前向传播，处理输入图像并产生编码后的图像，image是输入张量
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # 输入图像张量然后输出特征图 (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  #使用自适应平均池化层将特征图的空间维度（即宽度和高度）转换为预定义的 size (batch_size, 2048, encoded_image_size, encoded_image_size)
        #out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048) 保持特定维度顺序，这个是后续attention layer工作所必需的

        out = out.view(out.size(0), out.size(1), -1)  # 转换为(batch_size, 2048, num_pixels)
        out = out.permute(0, 2, 1)  # 转换为(batch_size, num_pixels, 2048)

        # 应用 GCN
        gcn_out = self.gcn(out, self.adj_matrix)
        gcn_out = gcn_out.permute(0, 2, 1).view(out.size(0), 2048, self.enc_image_size, self.enc_image_size)

        return gcn_out.permute(0, 2, 3, 1)  # 转换为(batch_size, encoded_image_size, encoded_image_size, 2048)
    
    def create_adj_matrix(self, num_pixels):
        # 初始化邻接矩阵, 可以是固定的，也可以基于某种标准动态生成
        adj_matrix = torch.eye(num_pixels)
        return adj_matrix

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


# 添加GCN代码逻辑
1. ResNet-101 作为编码器
目的：使用ResNet-101是为了从输入图像中提取丰富的特征。ResNet-101是一个深度残差网络，经过预训练可以捕获图像的多层次抽象表示。
操作：你的Encoder类中首先使用了torchvision.models.resnet101，移除了最后的全连接层和池化层，因为我们对分类输出不感兴趣。我们只需要得到一个高维特征图。
输出：对于每个输入图像，Encoder输出一个(batch_size, 2048, 14, 14)的特征张量，然后通过自适应平均池化层和调整维度，得到(batch_size, 14, 14, 2048)的特征图。这一步保证了即使输入图像尺寸不同，输出特征图的空间维度也是固定的。
2. GCN 进一步提取特征
目的：GCN的引入是为了利用图像内部区域（节点）之间的空间关系，从而捕获更加丰富的上下文信息。GCN可以帮助模型理解不同图像区域（如物体和背景）之间的相互作用。
构建邻接矩阵：你首先根据每个图像的
14
×
14
14×14区域构建了一个邻接矩阵，表示这些区域（节点）之间的空间连接（基于8邻域连接）。这个邻接矩阵是固定的，因为每张图像都被处理成了相同的空间尺寸。
操作：GCNModule接收Encoder的输出，并首先调整维度以匹配GCN的输入要求，然后应用图卷积网络。由于每张图像的结构相同，所以同一批次中的所有图像可以使用相同的edge_index。在GCN处理之后，特征图被恢复到与Encoder输出相同的维度(batch_size, 14, 14, 2048)。
保持图像独立性：尽管我们在计算时将批次中的图像一起处理，但通过使用相同的edge_index并保持输出特征图的维度不变，我们确保了每张图像的特征提取过程是独立的。这意味着每张图像都有其对应的特征表示，可以与相应的字幕匹配，从而支持后续的图像字幕生成任务。

# Attention 和Lstm的输入输出参数记录
输入和输出：
Attention Module

输入：encoder_out, decoder_hidden
输出：attention_weighted_encoding, alpha
DecoderWithAttention Module

输入：encoder_out, encoded_captions, caption_lengths
输出：predictions, encoded_captions, decode_lengths, alphas, sort_ind