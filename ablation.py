import torch
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
# 数据集类
class MultimodalDataset(Dataset):
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

# 仅文本模型
class TextOnlyModel(nn.Module):
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
        
        self.classifier = nn.Sequential(
            nn.Linear(self.text_hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for layer in [-1, -2]:
            for param in self.text_encoder.encoder.layer[layer].parameters():
                param.requires_grad = True
    
    def forward(self, input_ids, attention_mask):
        text_features = self.text_encoder(input_ids, attention_mask).last_hidden_state
        
        text_attention = text_features
        for attention, norm in zip(self.text_attention_layers, self.text_layer_norms):
            attn_out, _ = attention(
                text_attention, text_attention, text_attention,
                key_padding_mask=~attention_mask.bool()
            )
            text_attention = norm(text_attention + attn_out)
        
        text_pooled = text_attention[:, 0]
        return self.classifier(text_pooled)

# 仅图像模型
class ImageOnlyModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.image_encoder = nn.Sequential(*list(mobilenet.children())[:-1])
        self.image_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_hidden_size = 1280
        
        self.classifier = nn.Sequential(
            nn.Linear(self.image_hidden_size, 1024),
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
    
    def _init_weights(self):
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in list(self.image_encoder.parameters())[-2:]:
            param.requires_grad = True
    
    def forward(self, image):
        image_features = self.image_encoder(image)
        image_features = self.image_pool(image_features)
        image_features = image_features.view(image_features.size(0), -1)
        return self.classifier(image_features)

def train_single_modality(model, train_loader, val_loader, modality, num_epochs=5, device='cuda'):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    if isinstance(model, TextOnlyModel):
        optimizer = torch.optim.AdamW([
            {'params': model.classifier.parameters(), 'lr': 1e-4},
            {'params': model.text_attention_layers.parameters(), 'lr': 2e-5},
            {'params': model.text_encoder.encoder.layer[-2:].parameters(), 'lr': 5e-6}
        ])
    else:  
        optimizer = torch.optim.AdamW([
            {'params': model.classifier.parameters(), 'lr': 1e-4},
            {'params': list(model.image_encoder.parameters())[-2:], 'lr': 5e-6}
        ])
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[1e-4, 5e-6] if isinstance(model, ImageOnlyModel) else [1e-4, 2e-5, 5e-6],
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            if modality == 'text':
                inputs = (batch['input_ids'].to(device), 
                         batch['attention_mask'].to(device))
            else: 
                inputs = batch['image'].to(device)
                
            labels = batch['label'].to(device)
            
            outputs = model(*inputs) if modality == 'text' else model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if modality == 'text':
                    inputs = (batch['input_ids'].to(device), 
                            batch['attention_mask'].to(device))
                else:  
                    inputs = batch['image'].to(device)
                    
                labels = batch['label'].to(device)
                
                outputs = model(*inputs) if modality == 'text' else model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
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
            }, f'best_{modality}_model.pth')
    
    history_filename = f'{modality}_training_history.json'
    with open(history_filename, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_filename}")
    
    return history

def plot_results(text_history, image_history):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(text_history['train_acc'], label='Text Train')
    plt.plot(text_history['val_acc'], label='Text Val')
    plt.plot(image_history['train_acc'], label='Image Train')
    plt.plot(image_history['val_acc'], label='Image Val')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(text_history['train_loss'], label='Text Train')
    plt.plot(text_history['val_loss'], label='Text Val')
    plt.plot(image_history['train_loss'], label='Image Train')
    plt.plot(image_history['val_loss'], label='Image Val')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ablation_results.png')
    plt.close()

def main():
    torch.manual_seed(1430)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
  
    data_dir = './P5_data/data'
    train_txt_path = './P5_data/train.txt'
    
    dataset = MultimodalDataset(text_path=train_txt_path, data_dir=data_dir)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    
    print("\nTraining Text-Only Model...")
    text_model = TextOnlyModel().to(device)
    text_history = train_single_modality(text_model, train_loader, val_loader, 'text', num_epochs=10, device=device)
    
    print("\nTraining Image-Only Model...")
    image_model = ImageOnlyModel().to(device)
    image_history = train_single_modality(image_model, train_loader, val_loader, 'image', num_epochs=10, device=device)
    
    plot_results(text_history, image_history)
    print("\nTraining completed. Results have been saved to 'ablation_results.png'")

if __name__ == '__main__':
    main()