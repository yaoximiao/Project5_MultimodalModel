import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModel, AutoTokenizer
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
# from torch.optim.lr_scheduler import CosineAnnealingLR

class MultimodalDataset(Dataset):
    # 一些初始化量，关键是对图片数据预处理，统一图片尺寸和对像素中心化，标准化
    def __init__(self, text_path, data_dir, transform=None, max_length=150):
        self.data = pd.read_csv(text_path, sep=',')
        self.data_dir = data_dir
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        self.label_map = {'positive': 0, 'neutral': 1, 'negative': 2}
    
    # 获取训练集数据的长度
    def __len__(self):
        return len(self.data)
    
    # 加载文本和图像并编码，最终返回字典[guid, 文本张量, 注意力掩码, 图片张量, 情感标签]
    def __getitem__(self, idx):
        guid = self.data.iloc[idx]['guid']
        
        guid = int(float(guid))
        text_path = os.path.join(self.data_dir, f"{guid}.txt")
        encodings = ['utf-8', 'gbk', 'latin1', 'iso-8859-1'] 
        text = None
        for enc in encodings:
            try:
                with open(text_path, 'r', encoding=enc) as f:
                    text = f.read().strip()
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            raise ValueError(f"无法解码文件: {text_path}")
        

        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        

        image_path = os.path.join(self.data_dir, f"{guid}.jpg")
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        

        if 'tag' in self.data.columns:
            label = self.label_map[self.data.iloc[idx]['tag']]
            label = torch.tensor(label)
        else:
            label = torch.tensor(-1)

        return {
            'guid': guid,
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'image': image,
            'label': label
        }

class LightweightMultimodalModel(nn.Module):
    # 初始化模型结构，定义文本编码器、图像编码器、注意力机制和融合层。
    # - 使用 BERT 作为文本编码器，MobileNetV2 作为图像编码器。
    # - 定义文本和图像的多头注意力层，以及跨模态的注意力机制。
    # - 初始化特征融合层，将文本和图像特征结合并输出分类结果。
    def __init__(self, num_classes=3):
        super().__init__()

        self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')
        self.text_hidden_size = 768
        
        self.text_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.text_hidden_size,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            ) for _ in range(2) 
        ])
        self.text_layer_norms = nn.ModuleList([
            nn.LayerNorm(self.text_hidden_size) for _ in range(2)
        ])

        mobilenet = models.mobilenet_v2(pretrained=True)
        self.image_encoder = nn.Sequential(*list(mobilenet.children())[:-1])
        self.image_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_hidden_size = 1280
        
        self.image_enhancement = nn.Sequential(
            nn.Linear(self.image_hidden_size, self.text_hidden_size),
            nn.LayerNorm(self.text_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.img2text_attention = nn.MultiheadAttention(
            embed_dim=self.text_hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.text2img_attention = nn.MultiheadAttention(
            embed_dim=self.text_hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self.fusion = nn.Sequential(
            nn.Linear(self.text_hidden_size * 3, 1024), 
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        self._init_weights()
    
    # 冻结文本编码器（BERT）和图像编码器（MobileNetV2）的大部分参数，仅微调 BERT 的最后两层。
    # 这是为了减少计算量并防止过拟合，同时允许模型适应特定任务。
    def _init_weights(self):
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        for layer in [-1, -2]:
            for param in self.text_encoder.encoder.layer[layer].parameters():
                param.requires_grad = True

    # 定义模型的前向传播过程：
    # - 对输入文本进行编码，提取文本特征，并通过多头注意力层增强特征。
    # - 对输入图像进行编码，提取图像特征，并将其映射到与文本特征相同的维度。
    # - 使用跨模态注意力机制（图像到文本、文本到图像）进行特征交互。
    # - 将文本特征、图像特征和跨模态特征拼接，通过融合层输出分类结果。
    def forward(self, input_ids, attention_mask, image):
        text_features = self.text_encoder(input_ids, attention_mask).last_hidden_state
        
        text_attention = text_features
        for attention, norm in zip(self.text_attention_layers, self.text_layer_norms):
            attn_out, _ = attention(
                text_attention, text_attention, text_attention,
                key_padding_mask=~attention_mask.bool()
            )
            text_attention = norm(text_attention + attn_out)
        
        image_features = self.image_encoder(image)
        image_features = self.image_pool(image_features)
        image_features = image_features.view(image_features.size(0), -1)
        image_features = self.image_enhancement(image_features)
        image_features = image_features.unsqueeze(1)
        
        img2text, _ = self.img2text_attention(
            text_attention, image_features, image_features
        )
        text2img, _ = self.text2img_attention(
            image_features, text_attention, text_attention
        )
        
        text_pooled = text_attention[:, 0] 
        img2text_pooled = img2text[:, 0]
        text2img_pooled = text2img.squeeze(1)
        
        combined_features = torch.cat([
            text_pooled, img2text_pooled, text2img_pooled
        ], dim=1)
        
        return self.fusion(combined_features)

# 绘制训练和验证的损失和准确率
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracies'], label='Train Acc')
    plt.plot(history['val_accuracies'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

 
# 训练多模态模型的主函数，包括以下步骤：
# 1. 定义损失函数（带标签平滑的交叉熵损失）和优化器（AdamW）。
# 2. 使用 OneCycleLR 学习率调度器动态调整学习率。
# 3. 在每个 epoch 中：
#     - 训练阶段：通过梯度累积和梯度裁剪优化模型参数。
#     - 验证阶段：评估模型在验证集上的性能。
# 4. 记录训练和验证的损失及准确率，支持早停机制以防止过拟合。
# 5. 保存最佳模型和训练历史到文件，便于后续分析和使用。
def train_model(model, train_loader, val_loader, num_epochs=5, device='cuda'):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = torch.optim.AdamW([
        {'params': model.fusion.parameters(), 'lr': 1e-4},
        {'params': model.text_attention_layers.parameters(), 'lr': 2e-5},
        {'params': model.text2img_attention.parameters(), 'lr': 2e-5},
        {'params': model.img2text_attention.parameters(), 'lr': 2e-5},
        {'params': model.text_encoder.encoder.layer[-2:].parameters(), 'lr': 5e-6}
    ])
    
    # 学习率预热和余弦退火调度
    # warmup_epochs = 2  # 预热 2 个 epoch
    # total_steps = num_epochs * len(train_loader)
    # warmup_steps = warmup_epochs * len(train_loader)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[1e-4, 2e-5, 2e-5, 2e-5, 5e-6],  
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1, 
        div_factor=25.0, 
        final_div_factor=1e4
    )
    # scheduler = CosineAnnealingLR(
    #     optimizer,
    #     T_max=total_steps - warmup_steps,  # 余弦退火的周期
    #     eta_min=1e-7  # 最小学习率
    # )
    
    best_val_acc = 0
    patience = 3
    patience_counter = 0
    
    accumulated_steps = 0
    accumulation_steps = 4
    
    history = {
        'train_losses': [], 'train_accuracies': [],
        'val_losses': [], 'val_accuracies': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask, images)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            accumulated_steps += 1
            
            if accumulated_steps % accumulation_steps == 0:
                # 学习率预热
                # if epoch < warmup_epochs:
                #     for param_group in optimizer.param_groups:
                #         param_group['lr'] = param_group['initial_lr'] * (i + 1) / warmup_steps
                # else:
                #     scheduler.step()  # 余弦退火调度
                
                # optimizer.step()
                # optimizer.zero_grad()
                optimizer.step()
                scheduler.step() 
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            torch.cuda.empty_cache()

        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask, images)
                loss = criterion(outputs, labels)
            
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                torch.cuda.empty_cache()
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        history['train_losses'].append(avg_train_loss)
        history['train_accuracies'].append(train_acc)
        history['val_losses'].append(avg_val_loss)
        history['val_accuracies'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Learning rates: {[group["lr"] for group in optimizer.param_groups]}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history
            }, 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after epoch {epoch+1}')
                break
    
    # 保存训练历史到文件
    with open('training_history_cos&pre.json', 'w') as f:
        json.dump(history, f)
    
    return history

def get_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    )

def main():
    
    torch.manual_seed(1430)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    data_dir = './P5_data/data'
    train_txt_path = './P5_data/train.txt'
    
    
    dataset = MultimodalDataset(text_path=train_txt_path, data_dir=data_dir)
    
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    
    train_loader = get_dataloader(train_dataset, batch_size=16, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=16, shuffle=False)
    
    
    model = LightweightMultimodalModel(num_classes=3).to(device)
    
    history = train_model(model, train_loader, val_loader, num_epochs=10, device=device)
    plot_training_history(history) 

if __name__ == '__main__':
    main()