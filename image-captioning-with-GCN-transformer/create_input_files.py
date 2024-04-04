from utils import create_input_files
#该函数的目的是处理图像数据集和其对应的字幕，生成模型训练需要的输入文件。此为JSON格式。
if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='flickr8k',
                       karpathy_json_path = r'C:\Users\Bohan Zhang\Documents\GitHub\dataset\dataset_flickr8k.json', #指向包含训练集、验证集和测试集分割信息及字幕的JSON文件的路径。指向包含训练集、验证集和测试集分割信息及字幕的JSON文件的路径。

                       image_folder= r'C:\Users\Bohan Zhang\Documents\GitHub\dataset\flickr8k\Images',
                       captions_per_image=5, #每张图保留字幕数量
                       min_word_freq=5, #词汇表中词的最小出现频率，低于此频率的词会被视为未知词
                       output_folder= r'C:\Users\Bohan Zhang\Documents\GitHub\Graph-neural-networks-for-image-captioning\a-PyTorch-Tutorial-to-Image-Captioning-master\output_data',
                       max_len=50) #最大字幕长度
