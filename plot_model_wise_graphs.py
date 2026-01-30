import matplotlib.pyplot as plt

# Scores for each model
model_scores = {
    "DeepLab": (0.9799009835628454, 0.9607096615885429),
    "Attention_U-Net": (0.959, 0.923),
    "Ensemble": (0.9771260562682388, 0.9554046668916979),
    "ResNeXt101": (0.980000959722469, 0.9608973845542974)
}

for model, (dice, iou) in model_scores.items():
    plt.figure()

    metrics = ["Dice Score", "IoU Score"]
    values = [dice, iou]

    plt.bar(metrics, values)
    plt.ylim(0.9, 1.0)
    plt.title(f"{model} Performance")
    plt.ylabel("Score")

    plt.savefig(f"{model}_performance.png")
    plt.close()

print("All model-wise graphs created successfully.")

