import torch
import math
from torch import nn
import numpy as np
import torchvision
from torchvision.models.resnet import ResNet101_Weights
from torch_geometric.nn import GCNConv # 用于GCN的model
from torch.nn import TransformerDecoder, TransformerDecoderLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#adjacency_matrix   output is adj_matrix  shape [196,196]
def build_adjacency_matrix():
    adj_matrix = np.zeros((196, 196))  # 创建一个196x196的邻接矩阵
    for row in range(14):
        for col in range(14):
            index = row * 14 + col  # 将2D坐标转换为一维索引
            # 对于每个节点，检查其8邻域
            for i in range(max(0, row-1), min(row+2, 14)):
                for j in range(max(0, col-1), min(col+2, 14)):
                    if i == row and j == col:
                        continue  # 跳过自身
                    neighbor_index = i * 14 + j
                    adj_matrix[index, neighbor_index] = 1  # 标记为邻居
    return adj_matrix      

#Convert adjacency matrix to edge index  使用边索引而不是邻接矩阵可以在某些场景下降低内存占用
def adjacency_matrix_to_edge_index(adj_matrix):
    edge_index = np.array(adj_matrix.nonzero())
    return torch.tensor(edge_index, dtype=torch.long)

# 构建邻接矩阵
adj_matrix = build_adjacency_matrix()
# 转换为边索引    一共1404个边 shape[2,1404] 2是source to target   这个时候图里面的关系就转换成了edge_index
edge_index = adjacency_matrix_to_edge_index(adj_matrix)


#GCN module Input: (batch_size, encoded_image_size, encoded_image_size, 2048)  Output (batch_size, encoded_image_size, encoded_image_size, 2048) 
class GCNModule(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, edge_index, encoded_image_size=14, dropout_rate=0.5):
        super(GCNModule, self).__init__()
        self.in_features = in_features  # GCN输入特征维度
        self.out_features = in_features  # 为了保持和ResNet输出一致，这里输出特征维度与输入相同
        self.edge_index = edge_index  # 边索引
        self.encoded_image_size = encoded_image_size  # 编码后的图像尺寸
        # first layer of GCN，从in_features到hidden_features
        self.gcn1 = GCNConv(in_features, hidden_features)
        # Second layer of CN，从hidden_features到hidden_features
        self.gcn2 = GCNConv(hidden_features, out_features)
        self.relu = nn.ReLU()  # RelU
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):  #x is the output of Encoder. 缺少了邻接矩阵的输入呢！
        batch_size, H, W, C = x.size()  # x的维度(batch_size, 14, 14, 2048)
            #print(edge_index.shape)  #torch.Size([2, 1404])
            #print(adj_matrix.shape)   
            #print(x.shape) [200, 14, 14, 2048] 现在shape是正确的
        x = x.permute(0, 3, 1, 2).reshape(-1, C)  # 调整x的维度为(N, in_features)  #原来是view，改为reshape permute 了之后[batch_size, channels, height, width]; N是196乘以batch_size,也就是图
        #print(x.shape)  #现在的x size是这样的：torch.Size([39200, 2048]) batch size * 196

        # 依次通过两层GCN
        x = self.relu(self.gcn1(x, self.edge_index))  # 第一层GCN + ReLU; edge_index is here #torch.Size([39200, 1024])
        x = self.dropout(x)

        # 第二层GCN + ReLU（在最后一层后也加ReLU）
        x = self.relu(self.gcn2(x, self.edge_index))  #torch.Size([39200, 2048])
        x = self.dropout(x)

        # 将输出从GCN后的长列表转换回原始(batch_size, H, W, C)的形状
        x = x.reshape(batch_size, H, W, self.out_features).permute(0, 3, 1, 2)  # 首先变回(batch_size, 14, 14, out_features) #原来是view，改为reshape #torch.Size([200, 2048, 14, 14])
        x = x.permute(0, 2, 3, 1)  # 最后调整为(batch_size, 14, 14, 2048) 注意out_features应当对应2048

        return x

# Based on restnet 101 model   Input（Resize image） output: (batch_size, encoded_image_size, encoded_image_size, 2048) 
class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        #resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101 已经不能够使用了，需要改为weight
        resnet = torchvision.models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)  这个是encoder的输出！！
        return out

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

#Using Transformer Decoder
class ImageCaptioningModel(nn.Module):
    def __init__(self, feature_dim, embed_size, vocab_size, num_layers, num_heads, max_seq_length, dropout=0.5):
       
        """ feature_dim: 图像特征的维度。这是从图像中提取的特征向量的大小,通常由卷积神经网络CNN提取。
        embed_size: 嵌入层的维度。这是单词嵌入向量的大小,也是Transformer Decoder内部处理的固定大小的向量维度。
        vocab_size: 词汇表的大小。这代表模型能够生成的不同单词的总数，每个单词在模型中都有一个唯一的索引。
        num_layers: Transformer Decoder的层数。这决定了模型的深度,每一层都包含了一个自注意力机制和前馈神经网络
        num_heads: 多头注意力中的头数。在每个Transformer Decoder层中,多头注意力机制允许模型同时在不同的表示子空间上关注输入的不同部分。
        max_seq_length: 序列的最大长度。这是模型能够处理的最大单词数，用于限制生成的描述的长度。 """

        super().__init__()
        self.embed_size = embed_size  # 将embed_size保存为类的属性
        # 嵌入层，用于词汇的嵌入
        self.embed = nn.Embedding(vocab_size, embed_size)
        # Transformer的Decoder层
        self.transformer_decoder_layer = TransformerDecoderLayer(d_model=embed_size, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)
        # 线性层，用于从Transformer的特征转换为词汇空间
        self.fc = nn.Linear(embed_size, vocab_size)
        # 特征维度到嵌入维度的转换
        self.feature_to_embed = nn.Linear(feature_dim, embed_size)  #将图像特征从feature_dim维度转换到嵌入向量的维度embed_size。这使得可以将图像特征作为Transformer Decoder的一部分输入。
        # 位置编码
        self.pos_encoder = PositionalEncoding(embed_size, dropout, max_seq_length) #给嵌入向量添加位置信息
    
    def generate_square_subsequent_mask(self, sz):
        """生成一个sz x sz的后续掩码。"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, gcn_features, captions):
        # 将GCN的输出特征转换为嵌入维度
        #print(gcn_features.shape) #torch.Size([180, 14, 14, 2048])
        
        # 将特征图平铺成序列
        # 新的序列长度是 H*W，例如 14*14=196
        batch_size, H, W, channels = gcn_features.size()
        gcn_features_flat = gcn_features.view(batch_size, H*W, channels)  # [180, 196, 2048]

        # 如果需要，通过一个线性层调整channels维度以匹配embed_size
        # 假设 self.feature_to_embed 已经是一个适当的线性层
        gcn_features_processed = self.feature_to_embed(gcn_features_flat)  # [180, 196, embed_size]
        # 为了匹配Transformer的期望输入，需要将batch_size和序列长度维度调换
        gcn_features_processed = gcn_features_processed.permute(1, 0, 2)  # [196, 180, embed_size]
        #print(gcn_features_processed.shape) #torch.Size([196, 180, 512])

        # 嵌入词汇并加上位置编码
        embeds = self.embed(captions) * math.sqrt(self.embed_size)
        embeds = self.pos_encoder(embeds) #torch.Size([180, 51, 512])
        embeds = embeds.permute(1, 0, 2)  # 将 embeds 调整为 [seq_len, batch_size, embed_size]
        
        # 创建后续掩码
        target_seq_len = captions.size(1)  # 获取目标序列的长度
        subsequent_mask = self.generate_square_subsequent_mask(target_seq_len).to(captions.device)

        # 通过Transformer Decoder处理，并应用后续掩码
        transformer_output = self.transformer_decoder(tgt=embeds, memory=gcn_features_processed, tgt_mask=subsequent_mask)

        # 转换为词汇空间
        output = self.fc(transformer_output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)