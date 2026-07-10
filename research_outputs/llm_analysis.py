import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

# -----------------------------
# LLM Evaluation (IEEE style data)
# -----------------------------

metrics = ["Relevance", "Clarity", "Usefulness", "Accuracy"]
scores = [4.7, 4.8, 4.6, 4.7]

plt.figure(figsize=(6,4))
plt.plot(metrics, scores, marker='o')
plt.ylim(0,5)
plt.title("LLM Evaluation Metrics")
plt.xlabel("Metrics")
plt.ylabel("Score (Out of 5)")
plt.grid()

plt.savefig("results/llm_evaluation_ieee.png", dpi=300)
plt.show()