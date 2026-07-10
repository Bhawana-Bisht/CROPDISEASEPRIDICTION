import matplotlib.pyplot as plt
import numpy as np

models = ['ResNet18', 'MobileNetV2', 'VGG16', 'EfficientNet-B0']

precision = [98.7, 99.7, 97.5, 100.0]
recall = [98.6, 99.6, 97.4, 100.0]
f1_score = [98.6, 99.6, 97.4, 100.0]

x = np.arange(len(models))
width = 0.25

plt.figure(figsize=(10,6))

plt.bar(x - width, precision, width,
        label='Precision', color='royalblue')

plt.bar(x, recall, width,
        label='Recall', color='orange')

plt.bar(x + width, f1_score, width,
        label='F1-Score', color='green')

plt.xlabel('Models', fontsize=12)
plt.ylabel('Score (%)', fontsize=12)
plt.title('Comparative Performance Analysis: Precision, Recall and F1-Score', fontsize=14)

plt.xticks(x, models)
plt.ylim(95, 101)

plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()

plt.savefig('results/precision_recall_f1_comparison.png', dpi=300)
plt.show()