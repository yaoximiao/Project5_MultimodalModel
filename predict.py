import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from torchvision import models, transforms
import pandas as pd
import os
from PIL import Image
from MP_2 import LightweightMultimodalModel  
class TestMultimodalDataset(Dataset):
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

    def __len__(self):
        return len(self.data)

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
            raise ValueError(f"Unable to decode file: {text_path}")

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

        return {
            'guid': guid,
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'image': image
        }

def predict(model, test_loader, device):
    model.eval()
    predictions = []
    guids = []
    
    label_map = {0: 'positive', 1: 'neutral', 2: 'negative'}
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            batch_guids = batch['guid']

            outputs = model(input_ids, attention_mask, images)
            _, predicted = outputs.max(1)
  
            batch_predictions = [label_map[pred.item()] for pred in predicted]
            
            predictions.extend(batch_predictions)
            guids.extend(batch_guids.tolist())

    return guids, predictions

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_file = './P5_data/test_without_label.txt'
    data_dir = './P5_data/data'
    model_path = 'best_model.pth'
    
    test_dataset = TestMultimodalDataset(text_path=test_file, data_dir=data_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = LightweightMultimodalModel(num_classes=3).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    guids, predictions = predict(model, test_loader, device)
    
    with open('./P5_data/predictions.txt', 'w', encoding='utf-8') as f:
        f.write('guid,tag\n')
        for guid, pred in zip(guids, predictions):
            f.write(f'{guid},{pred}\n')
    
    print(f"Predictions saved to predictions.txt")

if __name__ == '__main__':
    main()