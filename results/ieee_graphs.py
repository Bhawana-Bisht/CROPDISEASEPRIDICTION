import matplotlib.pyplot as plt

# -----------------------------
# GLOBAL STYLE (IEEE LOOK)
# -----------------------------
plt.style.use('default')

# =============================
# 1. MODEL ACCURACY
# =============================
models = ['ResNet18','MobileNetV2','VGG16','EfficientNetB0']
accuracy = [98.9, 99.8, 97.9, 100.0]

plt.figure(figsize=(8,5))
bars = plt.bar(models, accuracy)

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

for bar, color in zip(bars, colors):
    bar.set_color(color)

plt.title("Model Accuracy Comparison (CNN Models)")
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.ylim(90, 101)

plt.savefig("results/1_model_accuracy.png", dpi=300)
plt.show()

# =============================
# 2. F1 SCORE
# =============================
f1 = [98.6, 99.6, 97.4, 100.0]

plt.figure(figsize=(8,5))
bars = plt.bar(models, f1)

for bar, color in zip(bars, colors):
    bar.set_color(color)

plt.title("F1 Score Comparison")
plt.xlabel("Models")
plt.ylabel("F1 Score (%)")
plt.ylim(90, 101)

plt.savefig("results/2_f1_score.png", dpi=300)
plt.show()

# =============================
# 3. PRECISION
# =============================
precision = [98.7, 99.7, 97.5, 100.0]

plt.figure(figsize=(8,5))
bars = plt.bar(models, precision)

for bar, color in zip(bars, colors):
    bar.set_color(color)

plt.title("Precision Comparison")
plt.xlabel("Models")
plt.ylabel("Precision (%)")
plt.ylim(90, 101)

plt.savefig("results/3_precision.png", dpi=300)
plt.show()

# =============================
# 4. RECALL
# =============================
recall = [98.6, 99.6, 97.4, 100.0]

plt.figure(figsize=(8,5))
bars = plt.bar(models, recall)

for bar, color in zip(bars, colors):
    bar.set_color(color)

plt.title("Recall Comparison")
plt.xlabel("Models")
plt.ylabel("Recall (%)")
plt.ylim(90, 101)

plt.savefig("results/4_recall.png", dpi=300)
plt.show()

# =============================
# 5. AUC SCORE
# =============================
auc = [0.995, 0.998, 0.991, 1.000]

plt.figure(figsize=(8,5))
bars = plt.bar(models, auc)

for bar, color in zip(bars, colors):
    bar.set_color(color)

plt.title("AUC Score Comparison")
plt.xlabel("Models")
plt.ylabel("AUC")
plt.ylim(0.95, 1.01)

plt.savefig("results/5_auc.png", dpi=300)
plt.show()

# =============================
# 6. NLP PERFORMANCE
# =============================
nlp_labels = ['Keyword Accuracy', 'Intent Accuracy']
nlp_values = [92, 94]

plt.figure(figsize=(6,5))
bars = plt.bar(nlp_labels, nlp_values, color=['#9b59b6', '#1abc9c'])

plt.title("NLP Module Performance")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

plt.savefig("results/6_nlp.png", dpi=300)
plt.show()

# =============================
# 7. HYBRID SYSTEM
# =============================
components = ['CNN Model','NLP Module','Overall System']
scores = [100, 94, 98.5]

plt.figure(figsize=(7,5))
bars = plt.bar(components, scores, color=['#e74c3c', '#9b59b6', '#2ecc71'])

plt.title("Hybrid System Performance")
plt.ylabel("Score (%)")
plt.ylim(0, 100)

plt.savefig("results/7_hybrid.png", dpi=300)
plt.show()