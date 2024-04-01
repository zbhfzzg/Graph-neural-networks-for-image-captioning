import torch
from torch import nn
import numpy as np
import torchvision
from torchvision.models.resnet import ResNet101_Weights
from torch_geometric.nn import GCNConv # 用于GCN的model
import torch_geometric.data as Data

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

#Attention 
class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):   #调用代码是：self.attention = Attention(encoder_dim, decoder_dim, attention_dim) 
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim) 问题是 GCN输出确定到了Attention layer了吗, 且要保证维度是(batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim) encoder_out通过一个线性层得到的，意在将编码后的图像特征投影到一个新的空间（即attention_dim）。
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim) 是将decoder_hidden通过一个线性层得到的，将解码器隐状态投影到相同的新空间。
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels) 是将att1和att2相加后通过ReLU激活函数，再经过一个线性层得到的，用来计算每个像素的注意力得分。
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim) 是将encoder_out与alpha相乘并求和得到的，代表了加权后的编码器输出，将被用作解码器的输入。

        return attention_weighted_encoding, alpha

# LSTM + Attention
class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network  是这行代码，让Attention model工作

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind] #当前时间步的输入词语的编码。

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
