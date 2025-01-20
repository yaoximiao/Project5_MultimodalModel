# Multimodal Model

此仓库包含一个轻量化的多模态情感分类模型

## Setup

该模型的主要依赖如下：

- torch==1.11.0+cu115

- scikit-learn==1.6.0

- transformers==4.47.1

- torchversion==0.12.0+cu115

- numpy==1.23.5

- pandas==2.2.3

- matplotlib==3.10.0

- seaborn==0.13.2

- PIL==11.1.0

- tqdm==4.67.1

由于本地显卡的特殊性，上述版本的torch和torchversion可能无法直接下载，需要到官网另外找寻，可以根据自己的设备更换版本，后续运行以下命令即可：

```python
pip install -r requirements.txt
```

## Repository structure
主要的文件目录如下

```python
|-- helper #一些工具类
    |-- picture_size.py #查看图像尺寸
    |-- plot_ablation.py #绘制消融实验对比图
    |-- plot_attention.py #绘制添加注意力层前后对比图
    |-- plot_diff_image_models.py #绘制不同图像预处理模型对比图
    |-- plot_diff_models.py #绘制不同文本预处理模型对比图
    |-- plot_pre&cos.py #绘制使用学习率预热和余弦退火调度对比图
    |-- text_length.py  #查看文本尺寸
|-- history_json #训练过程数据
    |-- image_training_history.json #仅图形模型
    |-- MP_1_training_history_deberta.json #使用deberta训练
    |-- MP_1_training_history_EfficientNet.json #使用EfficientNet训练
    |-- MP_1_training_history_roberta.json #使用roberta训练
    |-- MP_2_training_history.json #最终的多模态模型
    |-- text_training_history.json #仅文本模型
    |-- training_history_cos&pre.json #使用学习率预热和余弦退火调度
|-- P5_data
    |-- data #包含图像和文本数据对
    |-- test_without_label.txt #测试集，仅有guid，没有tag
    |-- train.txt #训练集，包含guid和tag
    |-- predictions.txt #测试集结果
|-- test_image #一些图像素材
    |-- 架构图.drawio
    |-- ablation.png
    |-- base&attention.png
    |-- compare_diff_image_models.png
    |-- compare_diff_text_models.png
    |-- pre&cos.png
    |-- training_history.png
|-- 实验五要求.pptx
|-- ablation.py #消融实验代码
|-- best_image_model.pth
|-- best_model.pth
|-- best_text_model.pth
|-- MP_1.py #基础版本的多模态模型代码
|-- MP_2.py #优化后的多模态模型代码
|-- MP_3_progressive_unfreezing.py #尝试逐层解冻的代码
|-- predict.py #预测测试集tag
|-- readme.md
|-- requirements.txt #依赖的库文件
```

## Run
进入主文件夹后直接运行以下代码即可实现对模型的训练过程
```python
python MP_2.py
```

运行以下代码则是对测试集的预测
```python
python predict.py
```
结果文件会生成在./P5_data/predictions.txt中

## Attribution
该多模式情感分类模型建立在几个主要库和预训练模型之上：

- 文本编码器：来自Hugging Face Transformers的BERT 

- 图像编码器：PyTorch torchvision中的MobileNetV2 深度学习框架：PyTorch

BERT用于文本特征提取，MobileNetV2用于图像特征提取，多头注意力机制用于跨模式特征融合，轻量级设计以实现高效的训练和推理。此实现旨在使用文本和图像输入进行情感分类任务，输出积极、中立或消极预测。

