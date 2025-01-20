import torch
import torch.nn as nn
import torch.nn.functional as F
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
import math
from tqdm import tqdm

class MultimodalDataset(Dataset):
    def __init__(self, text_path, data_dir, transform=None, max_length=150):
        self.data = pd.read_csv(text_path, sep=',')
        self.data_dir = data_dir
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        self.label_map = {'positive': 0, 'neutral': 1, 'negative': 2}
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        guid = self.data.iloc[idx]['guid']
        
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

class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class EnhancedMultimodalModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
     
        self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')
        self.text_hidden_size = 768
    
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.image_encoder = nn.Sequential(*list(self.mobilenet.children())[:-1])
        self.se = SEModule(1280)
        self.image_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_hidden_size = 1280
        
        self.text_attention = nn.MultiheadAttention(
            self.text_hidden_size, 8, dropout=0.1, batch_first=True
        )
        self.image_attention = nn.MultiheadAttention(
            self.text_hidden_size, 8, dropout=0.1, batch_first=True
        )
        
        self.image_projection = nn.Sequential(
            nn.Linear(self.image_hidden_size, self.text_hidden_size),
            nn.LayerNorm(self.text_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.text_norm = nn.LayerNorm(self.text_hidden_size)
        self.image_norm = nn.LayerNorm(self.text_hidden_size)
        
        self.fusion = nn.Sequential(
            nn.Linear(self.text_hidden_size * 3, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        for layer in [-1, -2]:
            for param in self.text_encoder.encoder.layer[layer].parameters():
                param.requires_grad = True
    
    def forward(self, input_ids, attention_mask, image):
        text_output = self.text_encoder(input_ids, attention_mask)
        text_features = text_output.last_hidden_state
        
        image_features = self.image_encoder(image)
        image_features = self.se(image_features)
        image_features = self.image_pool(image_features)
        image_features = image_features.view(image_features.size(0), -1)
        image_features = self.image_projection(image_features)
        image_features = image_features.unsqueeze(1)

        sequence_length = attention_mask.sum(dim=1)  
        key_padding_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        for i, length in enumerate(sequence_length):
            key_padding_mask[i, length:] = True 

        text_attended, _ = self.text_attention(
            text_features, image_features, image_features,
            key_padding_mask=key_padding_mask
        )
        image_attended, _ = self.image_attention(
            image_features, text_features, text_features,
            key_padding_mask=key_padding_mask
        )

        text_features = self.text_norm(text_features + text_attended)
        image_features = self.image_norm(image_features + image_attended)

        text_pooled = text_features[:, 0]
        text_attended_pooled = text_attended[:, 0]
        image_pooled = image_features.squeeze(1)
        
        combined = torch.cat([text_pooled, text_attended_pooled, image_pooled], dim=1)
        output = self.fusion(combined)
        
        return output
    # def forward(self, input_ids, attention_mask, image):
    #     # Text Features
    #     text_output = self.text_encoder(input_ids, attention_mask)
    #     text_features = text_output.last_hidden_state
        
    #     # Image Features
    #     image_features = self.image_encoder(image)
    #     image_features = self.se(image_features)
    #     image_features = self.image_pool(image_features)
    #     image_features = image_features.view(image_features.size(0), -1)
    #     image_features = self.image_projection(image_features)
    #     image_features = image_features.unsqueeze(1)
        
    #     # Cross Attention
    #     text_attended, _ = self.text_attention(
    #         text_features, image_features, image_features,
    #         key_padding_mask=~attention_mask.bool()
    #     )
    #     image_attended, _ = self.image_attention(
    #         image_features, text_features, text_features,
    #         key_padding_mask=~attention_mask.bool()
    #     )
        
    #     # Feature Normalization
    #     text_features = self.text_norm(text_features + text_attended)
    #     image_features = self.image_norm(image_features + image_attended)
        
    #     # Feature Pooling
    #     text_pooled = text_features[:, 0]
    #     text_attended_pooled = text_attended[:, 0]
    #     image_pooled = image_features.squeeze(1)
        
    #     # Fusion
    #     combined = torch.cat([text_pooled, text_attended_pooled, image_pooled], dim=1)
    #     output = self.fusion(combined)
        
    #     return output

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    criterion = FocalLoss(gamma=2.0)
    
    optimizer = torch.optim.AdamW([
        {'params': model.fusion.parameters(), 'lr': 1e-4},
        {'params': model.text_attention.parameters(), 'lr': 5e-5},
        {'params': model.image_attention.parameters(), 'lr': 5e-5},
        {'params': model.text_encoder.encoder.layer[-2:].parameters(), 'lr': 1e-5}
    ])
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[1e-4, 5e-5, 5e-5, 1e-5],
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    best_val_acc = 0
    patience = 3
    patience_counter = 0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask, images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
            
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
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
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

    with open('training_history_MP_3.json', 'w') as f:
        json.dump(history, f)
    
    return history

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def main():
    torch.manual_seed(1430)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_dir = './P5_data/data'
    train_txt_path = './P5_data/train.txt'

    dataset = MultimodalDataset(text_path=train_txt_path, data_dir=data_dir)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = EnhancedMultimodalModel(num_classes=3).to(device)
    
    history = train_model(model, train_loader, val_loader, num_epochs=10, device=device)
    
    plot_training_history(history)

if __name__ == "__main__":
    main()
