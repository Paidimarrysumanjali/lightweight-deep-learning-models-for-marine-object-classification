import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import itertools
import matplotlib.pyplot as plt
import time
import pandas as pd # Import pandas for table generation
# from torchsummary import summary # Uncomment if you want to use summary here for FLOPs/Params directly

# --- 1. Configuration ---
# Adjusted paths for local VS Code execution (relative to project root)
DATA_DIR = 'dataset'
MODEL_SAVE_DIR = 'models'
BATCH_SIZE = 32
NUM_CLASSES = 5
CLASS_NAMES = ['dolphin', 'fish', 'lobster', 'octopus', 'sea_horse']
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- 2. Data Preprocessing for Evaluation ---
data_transforms_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Loading test data...")
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms_test)
                  for x in ['test']} # Only load test data for evaluation
# Set num_workers=0 for better compatibility on Windows/VS Code
test_dataloader = DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_dataset_size = len(image_datasets['test'])

# Verify class names order
print(f"Detected class names: {image_datasets['test'].classes}")
if not all(a == b for a, b in zip(CLASS_NAMES, image_datasets['test'].classes)):
    print("WARNING: Class names order mismatch in evaluate_models.py. Using dataset order.")
    CLASS_NAMES = image_datasets['test'].classes

# --- 3. Model Definitions (MUST match train_models.py for structure) ---
# --- (These functions remain identical to your train_models.py and app/app.py) ---

def load_mobilenetv2_model():
    # Use weights=... for newer torchvision, or pretrained=True for older
    try:
        model = models.mobilenet_v2(weights=models.MobileNetV2_Weights.IMAGENET1K_V1)
    except AttributeError:
        # If weights enum not found (older torchvision), try pretrained=False for structure
        model = models.mobilenet_v2(pretrained=False)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, 'mobilenetv2_marine_classifier.pth'), map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

def load_efficientnetb0_model():
    # Use weights=... for newer torchvision, or pretrained=True for older
    try:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    except AttributeError:
        model = models.efficientnet_b0(pretrained=False) # Only structure
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, 'efficientnetb0_marine_classifier.pth'), map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

class CustomCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Linear(64, num_classes))
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def load_customcnn_model():
    model = CustomCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, 'customcnn_marine_classifier.pth'), map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

# --- 4. Evaluation Function ---
def evaluate_model(model, dataloader, model_name, num_inference_samples=100):
    if model is None:
        print(f"\n--- Skipping {model_name} evaluation (model not loaded) ---")
        return None, None

    print(f"\n--- Evaluating {model_name} ---")
    corrects = 0
    all_labels = []
    all_preds = []
    inference_times = []

    # Ensure at least one batch is processed for timing if num_inference_samples is small
    num_batches_to_sample = max(1, num_inference_samples // dataloader.batch_size)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # Measure inference time for a few samples to get an average
            if i < num_batches_to_sample:
                start_time = time.time()
                outputs = model(inputs)
                end_time = time.time()
                # Time for batch, divide by batch size to get time per image
                inference_times.append((end_time - start_time) / inputs.size(0))
            else: # For the rest, just do inference without timing overhead
                outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = corrects.double() / test_dataset_size
    avg_inference_time_ms = (np.mean(inference_times) * 1000) if inference_times else 0

    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Average Inference Time (per image): {avg_inference_time_ms:.2f} ms')

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, CLASS_NAMES, title=f'Confusion Matrix for {model_name}')

    return accuracy.item(), avg_inference_time_ms

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, fmt.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def plot_bar_chart(model_names, values, title, ylabel, color='skyblue'):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, values, color=color)
    plt.xlabel('Model')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(bottom=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        # Adjust text position based on value to prevent overlap with bar top/bottom
        if yval > 0.05 * plt.ylim()[1]: # Place above bar if value is significant
             plt.text(bar.get_x() + bar.get_width()/2, yval + (yval*0.02), round(yval, 2), ha='center', va='bottom')
        else: # Place inside bar if very small
            plt.text(bar.get_x() + bar.get_width()/2, yval + (plt.ylim()[1]*0.01), round(yval, 2), ha='center', va='bottom', color='black') # Adjusted text color for small bars

    plt.tight_layout()
    plt.show()


# --- 5. Main Evaluation Loop ---
if __name__ == '__main__':
    all_models = {
        "MobileNetV2": None,
        "EfficientNet-B0": None,
        "Custom CNN": None
    }

    # --- Static Model Metrics for table and plotting (YOUR PROVIDED VALUES) ---
    # GFLOPs are estimated or from torchsummary output, as you provided Parameters.
    # Accuracy and Inference Time will be calculated by the script.
    MODEL_STATIC_METRICS = {
        "MobileNetV2": {"Parameters (Millions)": 2.23, "GFLOPs": 0.33},
        "EfficientNet-B0": {"Parameters (Millions)": 4.01, "GFLOPs": 0.39},
        "Custom CNN": {"Parameters (Millions)": 0.02, "GFLOPs": 0.01},
    }

    # Load models
    for model_key in all_models.keys():
        try:
            if model_key == "MobileNetV2":
                all_models[model_key] = load_mobilenetv2_model()
            elif model_key == "EfficientNet-B0":
                all_models[model_key] = load_efficientnetb0_model()
            elif model_key == "Custom CNN":
                all_models[model_key] = load_customcnn_model()
        except FileNotFoundError:
            print(f"{model_key} weights not found. Ensure train_models.py was run.")
        except Exception as e:
            print(f"Error loading {model_key}: {e}")

    # Collect all results dynamically and statically
    model_results_list = []

    for model_name, model in all_models.items():
        if model is not None:
            acc, infer_time = evaluate_model(model, test_dataloader, model_name)
            if acc is not None and infer_time is not None:
                # Combine dynamic results with static metrics
                result = {
                    'Model': model_name,
                    'Accuracy (%)': round(acc * 100, 2), # Convert to percentage
                    'Inference Time (ms/image)': round(infer_time, 2),
                    'Parameters (Millions)': MODEL_STATIC_METRICS[model_name]["Parameters (Millions)"],
                    'GFLOPs': MODEL_STATIC_METRICS[model_name]["GFLOPs"]
                }
                model_results_list.append(result)

    # --- Generate Metrics Table ---
    if model_results_list:
        print("\n--- Comparative Model Performance Metrics ---")
        metrics_df = pd.DataFrame(model_results_list)
        metrics_df = metrics_df.set_index('Model') # Set 'Model' as index for cleaner display
        print(metrics_df.to_string()) # Use to_string() to print full DataFrame without truncation
        print("-" * 50)

        # --- Plotting the comparative results ---
        # Extract data for plotting from the DataFrame
        model_names_for_plots = metrics_df.index.tolist()
        accuracies_for_plots = metrics_df['Accuracy (%)'].tolist()
        inference_times_for_plots = metrics_df['Inference Time (ms/image)'].tolist()
        params_for_plots = metrics_df['Parameters (Millions)'].tolist()
        flops_for_plots = metrics_df['GFLOPs'].tolist()

        plot_bar_chart(model_names_for_plots, accuracies_for_plots, 'Model Test Accuracy', 'Accuracy (%)', color='lightgreen')
        plot_bar_chart(model_names_for_plots, inference_times_for_plots, 'Average Inference Time (per image)', 'Time (ms)', color='lightcoral')
        plot_bar_chart(model_names_for_plots, params_for_plots, 'Model Parameters', 'Parameters (Millions)', color='skyblue')
        plot_bar_chart(model_names_for_plots, flops_for_plots, 'Model GFLOPs', 'GFLOPs', color='gold')
    else:
        print("\nNo models were successfully loaded or evaluated to generate tables and plots.")