import time
import os
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import *
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from torch.utils.data import DataLoader

# Data parameters
data_folder = r'C:\\Users\\Bohan Zhang\\Documents\\GitHub\\Graph-neural-networks-for-image-captioning\\a-PyTorch-Tutorial-to-Image-Captioning-master\\output_data'  # folder with data files saved by create_input_files.py
data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'  # base name shared by data files

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 20  # number of epochs to train for (if early stopping is not triggered) #原为120
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 120  #原来是32
workers = 0  # for data-loading; right now, only 1 works with h5py    #把workers改为0了
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none 原来问题出在你这里！

# 指定保存生成字幕的文件的路径
output_dir = r'C:\Users\Bohan Zhang\Documents\GitHub\Graph-neural-networks-for-image-captioning\a-PyTorch-Tutorial-to-Image-Captioning-master\output_data'
output_file_path = os.path.join(output_dir, 'generated_captions.txt')
# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    # 构建邻接矩阵并转换为边索引
    adj_matrix = build_adjacency_matrix()
    edge_index = adjacency_matrix_to_edge_index(adj_matrix).to(device)  # 确保edge_index在正确的设备上

    if checkpoint is None:
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None
        image_captioning_model = ImageCaptioningModel(
            feature_dim=2048,  # 假设这是从编码器得到的特征维度
            embed_size=emb_dim, #512
            vocab_size=len(word_map), #2633
            num_layers=1,
            num_heads=8,
            max_seq_length=196, #设置为14*14
            dropout=dropout
            ).to(device)
    
        # 初始化优化器
        image_captioning_model_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, image_captioning_model.parameters()),
            lr=decoder_lr  # 根据需要设置学习率
        )
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None
        # 实例化GCNModule
        gcn_module = GCNModule(in_features=2048, out_features=2048, hidden_features=1024, edge_index=edge_index, encoded_image_size=14).to(device)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        image_captioning_model = checkpoint['image_captioning_model']
        image_captioning_model_optimizer = checkpoint['image_captioning_model_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        
        # 加载GCNModule，如果保存在checkpoint中
        gcn_module = checkpoint.get('gcn_module')
        if gcn_module is None:
            # 如果checkpoint中没有GCNModule，则需要重新初始化
            gcn_module = GCNModule(in_features=2048, out_features=2048, hidden_features=1024, edge_index=edge_index, encoded_image_size=14).to(device)
        
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    # Move to GPU, if available
    image_captioning_model = image_captioning_model.to(device)
    encoder = encoder.to(device)
    gcn_module = gcn_module.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)   #把num_workers=workers 改成了 num_workers = 0
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              image_captioning_model=image_captioning_model,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              image_captioning_model_optimizer=image_captioning_model_optimizer,
              epoch=epoch,
              gcn_module=gcn_module)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                image_captioning_model=image_captioning_model,
                                criterion=criterion,
                                device=device,
                                encoder=encoder,
                                gcn_module=gcn_module)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, image_captioning_model, encoder_optimizer,
                        image_captioning_model_optimizer, gcn_module, recent_bleu4, is_best)


def train(train_loader, encoder, image_captioning_model, criterion, encoder_optimizer, image_captioning_model_optimizer, epoch, gcn_module):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    
    :param epoch: epoch number
    :param gcn_module: GCN
    """

    image_captioning_model.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):    #caps是一个批次内所有图像对应的字幕
        
        data_time.update(time.time() - start)
        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop through Encoder and GCNModule
        imgs = encoder(imgs)  # Encoder for extracting image features
        imgs = gcn_module(imgs)  # GCNModule for processing image features
        
        #print(caps.shape) torch.Size([180, 52]) 也就是字幕长度为52个词
        # Forward prop through ImageCaptioningModel
        # 注意：这里假设image_captioning_model的forward方法接收经过GCN处理的图像特征和去除最后一个词的字幕
        scores = image_captioning_model(imgs, caps[:, :-1])   #torch.Size([51, 180, 2633]) scores张量包含了对每个时间步的每个单词的预测分数，形状为[seq_len, batch_size, vocab_size]。要生成字幕，你需要从这个张量中为每个时间步选择一个单词

        # 由于Transformer模型能够处理变长序列，这里不需要使用pack_padded_sequence
        # 直接计算目标和损失
        targets = caps[:, 1:]  # 目标是去除开始标记的字幕  torch.Size([180, 51])
        loss = criterion(scores.reshape(-1, scores.size(-1)), targets.reshape(-1))
        
        """ # 清零图像标题模型的梯度并更新
        image_captioning_model_optimizer.zero_grad()  # 同样，清零操作应该在loss.backward()之后，但在optimizer.step()之前
        loss.backward()  # 执行反向传播一次，计算所有参与计算的参数的梯度
        image_captioning_model_optimizer.step()  # 更新图像标题模型的参数 """

        # 如果存在，清零编码器的梯度
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()

        # 清零图像标题模型的梯度
        image_captioning_model_optimizer.zero_grad()

        # 执行反向传播，计算梯度
        loss.backward()

        # 更新图像标题模型的参数
        image_captioning_model_optimizer.step()

        # 如果存在，更新编码器的参数
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # 假设 word_map['<pad>'] 代表填充标记的索引
        pad_index = word_map['<pad>']
        # 计算非填充标记的数量作为每个序列的实际长度
        decode_lengths = (caps != pad_index).sum(dim=1).cpu().numpy()
        
        """  # 从scores获得最高得分的索引
        _, predicted_indices = scores.max(dim=-1)#[51, 180]
        predicted_indices = predicted_indices.transpose(0, 1)  # 转置以匹配 targets 的形状 [180, 51] """
        
        # 计算并更新指标
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))  # 这里假设你有某种方式计算或存储decode_lengths
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # 打印训练进度信息
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                        batch_time=batch_time,
                                                                        data_time=data_time, loss=losses,
                                                                        top5=top5accs))


def validate(val_loader, image_captioning_model, criterion, device, encoder, gcn_module):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param image_captioning_model: Integrated model for image captioning using Transformer.
    :param criterion: loss layer
    :param device: Device to perform validation on (CPU or GPU).
    :param word_map: word mapping to convert indices to words.
    :param output_file_path: path to save generated captions.
    :return: BLEU-4 score
    """
    
    image_captioning_model.eval()  # Set model to evaluate mode
    rev_word_map = {v: k for k, v in word_map.items()}  # Index to word mapping

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = []  # True captions
    hypotheses = []  # Predicted captions

    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
            imgs, caps = imgs.to(device), caps.to(device)

            imgs = encoder(imgs)  # 使用encoder提取特征
            imgs = gcn_module(imgs)  # 使用gcn_module进一步处理特征
            # Forward pass
            scores = image_captioning_model(imgs, caps[:, :-1])

            # Compute loss
            targets = caps[:, 1:]  # Exclude the <start> token
            loss = criterion(scores.reshape(-1, scores.size(-1)), targets.reshape(-1))

            # Update metrics
            losses.update(loss.item(), imgs.size(0))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, imgs.size(0))
            batch_time.update(time.time() - start)

            # Prepare references and hypotheses for BLEU-4 evaluation
            allcaps = allcaps.cpu().numpy()
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: ' '.join([rev_word_map[word] for word in c if word not in {word_map['<start>'], word_map['<pad>'], word_map['<end>']}]),
                        img_caps))  # Convert back to words and remove special tokens
                references.append(img_captions)

            # Convert scores to predicted indices
            _, preds = torch.max(scores.detach().cpu(), dim=2)
            preds = preds.tolist()
            # Convert predicted indices to words
            temp_preds = list()
            for p in preds:
                temp_preds.append(' '.join([rev_word_map[word] for word in p if word not in {word_map['<start>'], word_map['<pad>'], word_map['<end>']}]))
            
            #这里添加了三行代码
            print(f"Number of hypotheses: {len(hypotheses)}")
            print(f"Number of references: {len(references)}")
            assert len(hypotheses) == len(references), "Mismatch in number of hypotheses and references"
            hypotheses.extend(temp_preds)

            start = time.time()

    # Calculate BLEU-4 score
    bleu4 = corpus_bleu(references, hypotheses)

    # Print and save the results
    print(f'\n * LOSS - {losses.avg:.3f}, TOP-5 ACCURACY - {top5accs.avg:.3f}, BLEU-4 - {bleu4}\n')
    with open(output_file_path, 'w') as file:
        for i, pred in enumerate(hypotheses):
            file.write(f"Caption {i+1}: {pred}\n")

    return bleu4

if __name__ == '__main__':
    main()