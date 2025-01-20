import json
import matplotlib.pyplot as plt

# 加载 JSON 文件
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# 加载三种模型的结果
image_results = load_json('./history_json/image_training_history.json')
text_results = load_json('./history_json/text_training_history.json')
MP_2_results = load_json('./history_json/MP_2_training_history.json')

# 提取数据
# 动态生成 epochs
def get_epochs(results):
    return range(1, len(results['train_losses']) + 1)

# 获取每个模型的 epochs
image_epochs = get_epochs(image_results)
text_epochs = get_epochs(text_results)
MP_2_epochs = get_epochs(MP_2_results)

# 绘制训练损失曲线
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.plot(image_epochs, image_results['train_losses'], label='only_image', marker='o')
plt.plot(text_epochs, text_results['train_losses'], label='only_text', marker='s')
plt.plot(MP_2_epochs, MP_2_results['train_losses'], label='image&text', marker='d')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True)

# 绘制验证损失曲线
plt.subplot(2, 2, 2)
plt.plot(image_epochs, image_results['val_losses'], label='only_image', marker='o')
plt.plot(text_epochs, text_results['val_losses'], label='only_text', marker='s')
plt.plot(MP_2_epochs, MP_2_results['val_losses'], label='image&text', marker='d')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss Comparison')
plt.legend()
plt.grid(True)

# 绘制训练准确率曲线
plt.subplot(2, 2, 3)
plt.plot(image_epochs, image_results['train_accuracies'], label='only_image', marker='o')
plt.plot(text_epochs, text_results['train_accuracies'], label='only_text', marker='s')
plt.plot(MP_2_epochs, MP_2_results['train_accuracies'], label='image&text', marker='d')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy Comparison')
plt.legend()
plt.grid(True)

# 绘制验证准确率曲线
plt.subplot(2, 2, 4)
plt.plot(image_epochs, image_results['val_accuracies'], label='only_image', marker='o')
plt.plot(text_epochs, text_results['val_accuracies'], label='only_text', marker='s')
plt.plot(MP_2_epochs, MP_2_results['val_accuracies'], label='image&text', marker='d')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()