import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModel, AutoTokenizer
from torchvision import models, transforms
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import json

class MultimodalDataset(Dataset):
    def __init__(self, text_path, data_dir, transform=None, max_length=64):
        self.data = pd.read_csv(text_path, sep=',')
        self.data_dir = data_dir
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        # self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        # self.tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base')
        self.max_length = max_length
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)), 
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

class LightweightMultimodalModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        
        # 文本特征提取器 (BERT)
        self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')
        # self.text_encoder = AutoModel.from_pretrained('roberta-base')
        # self.text_encoder = AutoModel.from_pretrained('microsoft/deberta-base')
        self.text_hidden_size = 768 
        
        # 图像特征提取器 (MobileNetV2)
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.image_encoder = nn.Sequential(*list(mobilenet.children())[:-1])
        # efficientnet = models.efficientnet_b0(pretrained=True)  # 使用 EfficientNet-B0
        # self.image_encoder = nn.Sequential(*list(efficientnet.children())[:-1]) 
        self.image_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_hidden_size = 1280  
        
        self.fusion = nn.Sequential(
            nn.Linear(self.text_hidden_size + self.image_hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        for param in self.text_encoder.encoder.layer[-1].parameters():
            param.requires_grad = True
            
    def forward(self, input_ids, attention_mask, image):
        with torch.no_grad():
            text_output = self.text_encoder(input_ids, attention_mask)
        text_features = text_output.last_hidden_state[:, 0, :]  
        
        with torch.no_grad():
            image_features = self.image_encoder(image)
            image_features = self.image_pool(image_features) 
            image_features = image_features.view(image_features.size(0), -1)  
        
        combined_features = torch.cat([text_features, image_features], dim=1)
        output = self.fusion(combined_features)
        
        return output

def train_model(model, train_loader, val_loader, num_epochs=20, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.fusion.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  
    
    best_val_acc = 0
    accumulated_steps = 0
    accumulation_steps = 4
    patience = 3
    epochs_without_improvement = 0
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    training_history = {
        'train_losses': [],
        'train_accuracies': [],
        'val_losses': [],
        'val_accuracies': []
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
            accumulated_steps += 1
            
            if accumulated_steps % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            torch.cuda.empty_cache()
        
        train_acc = 100. * train_correct / train_total
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_acc)

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
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_acc)
        
        training_history['train_losses'].append(train_loss / len(train_loader))
        training_history['train_accuracies'].append(train_acc)
        training_history['val_losses'].append(val_loss / len(val_loader))
        training_history['val_accuracies'].append(val_acc)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        scheduler.step()
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    
    plt.show()

    with open('MP_1_training_history_EfficientNet.json', 'w') as f:
        json.dump(training_history, f, indent=4)
    print("训练历史已保存到 MP_1_training_history_deberta.json")

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
    
    train_model(model, train_loader, val_loader, num_epochs=20, device=device)

if __name__ == '__main__':
    main()