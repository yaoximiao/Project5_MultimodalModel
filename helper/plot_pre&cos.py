import json
import matplotlib.pyplot as plt

# 加载 JSON 文件
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
base_results = load_json('./history_json/MP_2_training_history.json') 
pre_results = load_json('./history_json/training_history_cos&pre.json')  

# 动态生成 epochs
def get_epochs(results):
    return range(1, len(results['train_losses']) + 1)

# 获取每个模型的 epochs
mobilenet_epochs = get_epochs(base_results)
efficientnet_epochs = get_epochs(pre_results)

# 绘制训练损失曲线
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.plot(mobilenet_epochs, base_results['train_losses'], label='base', marker='o')
plt.plot(efficientnet_epochs, pre_results['train_losses'], label='pre&cos', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True)

# 绘制验证损失曲线
plt.subplot(2, 2, 2)
plt.plot(mobilenet_epochs, base_results['val_losses'], label='base', marker='o')
plt.plot(efficientnet_epochs, pre_results['val_losses'], label='pre&cos', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss Comparison')
plt.legend()
plt.grid(True)

# 绘制训练准确率曲线
plt.subplot(2, 2, 3)
plt.plot(mobilenet_epochs, base_results['train_accuracies'], label='base', marker='o')
plt.plot(efficientnet_epochs, pre_results['train_accuracies'], label='pre&cos', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy Comparison')
plt.legend()
plt.grid(True)

# 绘制验证准确率曲线
plt.subplot(2, 2, 4)
plt.plot(mobilenet_epochs, base_results['val_accuracies'], label='base', marker='o')
plt.plot(efficientnet_epochs, pre_results['val_accuracies'], label='pre&cos', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()