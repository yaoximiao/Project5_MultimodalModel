import json
import matplotlib.pyplot as plt

# 加载 JSON 文件
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# 加载两种模型的结果
mobilenet_results = load_json('./history_json/MP_1_training_history.json')  # MobileNetV2 的结果
efficientnet_results = load_json('./history_json/MP_1_training_history_EfficientNet.json')  # EfficientNet 的结果

# 动态生成 epochs
def get_epochs(results):
    return range(1, len(results['train_losses']) + 1)

# 获取每个模型的 epochs
mobilenet_epochs = get_epochs(mobilenet_results)
efficientnet_epochs = get_epochs(efficientnet_results)

# 绘制训练损失曲线
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.plot(mobilenet_epochs, mobilenet_results['train_losses'], label='MobileNetV2', marker='o')
plt.plot(efficientnet_epochs, efficientnet_results['train_losses'], label='EfficientNet', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True)

# 绘制验证损失曲线
plt.subplot(2, 2, 2)
plt.plot(mobilenet_epochs, mobilenet_results['val_losses'], label='MobileNetV2', marker='o')
plt.plot(efficientnet_epochs, efficientnet_results['val_losses'], label='EfficientNet', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss Comparison')
plt.legend()
plt.grid(True)

# 绘制训练准确率曲线
plt.subplot(2, 2, 3)
plt.plot(mobilenet_epochs, mobilenet_results['train_accuracies'], label='MobileNetV2', marker='o')
plt.plot(efficientnet_epochs, efficientnet_results['train_accuracies'], label='EfficientNet', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy Comparison')
plt.legend()
plt.grid(True)

# 绘制验证准确率曲线
plt.subplot(2, 2, 4)
plt.plot(mobilenet_epochs, mobilenet_results['val_accuracies'], label='MobileNetV2', marker='o')
plt.plot(efficientnet_epochs, efficientnet_results['val_accuracies'], label='EfficientNet', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()