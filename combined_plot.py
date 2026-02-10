import matplotlib.pyplot as plt
import numpy as np

models = ["MobileNetV2", "EfficientNet-B0", "Custom CNN"]

accuracy = [86, 85, 47]
inference_time = [22.92, 18.97, 4.67]
parameters = [2.23, 4.01, 0.02]

x = np.arange(len(models))
width = 0.25

plt.figure(figsize=(10,6))

plt.bar(x - width, accuracy, width, label="Accuracy (%)")
plt.bar(x, inference_time, width, label="Inference Time (ms)")
plt.bar(x + width, parameters, width, label="Parameters (M)")

plt.xticks(x, models)
plt.ylabel("Value")
plt.title("Model Performance Comparison")
plt.legend()

plt.tight_layout()
plt.show()
