import json
import matplotlib.pyplot as plt

# 加载 JSON 文件
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# 加载三种模型的结果
bert_results = load_json('./history_json/MP_1_training_history.json')
roberta_results = load_json('./history_json/MP_1_training_history_roberta.json')
deberta_results = load_json('./history_json/MP_1_training_history_deberta.json')

# 提取数据
# 动态生成 epochs
def get_epochs(results):
    return range(1, len(results['train_losses']) + 1)

# 获取每个模型的 epochs
bert_epochs = get_epochs(bert_results)
roberta_epochs = get_epochs(roberta_results)
deberta_epochs = get_epochs(deberta_results)

# 绘制训练损失曲线
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.plot(bert_epochs, bert_results['train_losses'], label='BERT', marker='o')
plt.plot(roberta_epochs, roberta_results['train_losses'], label='RoBERTa', marker='s')
plt.plot(deberta_epochs, deberta_results['train_losses'], label='DeBERTa', marker='d')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True)

# 绘制验证损失曲线
plt.subplot(2, 2, 2)
plt.plot(bert_epochs, bert_results['val_losses'], label='BERT', marker='o')
plt.plot(roberta_epochs, roberta_results['val_losses'], label='RoBERTa', marker='s')
plt.plot(deberta_epochs, deberta_results['val_losses'], label='DeBERTa', marker='d')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss Comparison')
plt.legend()
plt.grid(True)

# 绘制训练准确率曲线
plt.subplot(2, 2, 3)
plt.plot(bert_epochs, bert_results['train_accuracies'], label='BERT', marker='o')
plt.plot(roberta_epochs, roberta_results['train_accuracies'], label='RoBERTa', marker='s')
plt.plot(deberta_epochs, deberta_results['train_accuracies'], label='DeBERTa', marker='d')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy Comparison')
plt.legend()
plt.grid(True)

# 绘制验证准确率曲线
plt.subplot(2, 2, 4)
plt.plot(bert_epochs, bert_results['val_accuracies'], label='BERT', marker='o')
plt.plot(roberta_epochs, roberta_results['val_accuracies'], label='RoBERTa', marker='s')
plt.plot(deberta_epochs, deberta_results['val_accuracies'], label='DeBERTa', marker='d')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()