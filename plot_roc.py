import os
import cv2
import numpy as np
from glob import glob
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from typing import List

# ============================================================================
# Configuration section. Adjust the following paths/parameters as needed.
# ============================================================================
MODEL_CHECKPOINT = "./model_results/Unet/FIVES/DiceLoss/checkpoint-epoch200.pth"  # Path to the *.pth checkpoint file
IMG_DIR = "../FIVESoriginal/test/image"   # Directory containing test RGB images
LABEL_DIR = "../FIVESoriginal/test/label" # Directory containing ground-truth mask images
OUTPUT_FIGURE = "roc_curve_unet.png"            # Where to save the resulting plot
IMAGE_SIZE = 2048                           # Spatial size used during training (images will be resized to IMAGE_SIZE×IMAGE_SIZE)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 'cuda' or 'cpu'
# ============================================================================

# ----------------------------------------------------------------------------
# Utility functions (taken from project codebase where possible)
# ----------------------------------------------------------------------------

def clahe_equalized(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE equalisation exactly as during training."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return bgr


def load_paths(img_dir: str, label_dir: str):
    imgs = sorted(glob(os.path.join(img_dir, "*")))
    masks = sorted(glob(os.path.join(label_dir, "*")))
    assert len(imgs) == len(masks), "Number of images and masks mismatch"
    return imgs, masks


def preprocess_image(path: str, size: int) -> torch.Tensor:
    """Read image from *path* and convert to a normalised tensor."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = clahe_equalized(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    img = img / 255.0  # normalise to 0-1
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
    return torch.from_numpy(img)


def preprocess_mask(path: str, size: int) -> torch.Tensor:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    mask = mask / 255.0
    mask = np.expand_dims(mask, axis=0).astype(np.float32)  # add channel dim
    return torch.from_numpy(mask)


# ----------------------------------------------------------------------------
# Build model and load checkpoint
# ----------------------------------------------------------------------------

import networks as models  # local package – contains Unet definition

# Instantiate model (we assume Unet was used – adapt if you use a different model)
model = models.Unet().to(DEVICE)
print("[1/5] Loading model checkpoint...", flush=True)
checkpoint = torch.load(MODEL_CHECKPOINT, map_location=DEVICE)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
print("[2/5] Model loaded and set to eval mode.", flush=True)
print("[3/5] Gathering file paths...", flush=True)

# ----------------------------------------------------------------------------
# Collect predictions and ground-truth labels
# ----------------------------------------------------------------------------

imgs, masks = load_paths(IMG_DIR, LABEL_DIR)
print(f"Found {len(imgs)} samples for ROC evaluation", flush=True)
print("[4/5] Starting inference over test set...", flush=True)

all_probs: List[np.ndarray] = []
all_labels: List[np.ndarray] = []

with torch.no_grad():
    for img_path, mask_path in tqdm(list(zip(imgs, masks)), desc="Inference", total=len(imgs)):
        img_tensor = preprocess_image(img_path, IMAGE_SIZE).unsqueeze(0).to(DEVICE)  # add batch dim
        mask_tensor = preprocess_mask(mask_path, IMAGE_SIZE).to(DEVICE)

        logits = model(img_tensor)
        if isinstance(logits, tuple):  # some architectures (e.g., WNet) return aux output
            logits = logits[-1]
        probs = torch.sigmoid(logits).cpu().squeeze().numpy().flatten()
        labels = mask_tensor.cpu().squeeze().numpy().flatten()

        all_probs.append(probs)
        all_labels.append(labels)

# Concatenate across samples
all_probs_arr = np.concatenate(all_probs)
all_labels_arr = np.concatenate(all_labels)

# ----------------------------------------------------------------------------
# Compute ROC curve & AUC
# ----------------------------------------------------------------------------
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(all_labels_arr.astype(np.uint8), all_probs_arr)
roc_auc = auc(fpr, tpr)
print("[5/5] Inference complete. Computing ROC and generating plot...", flush=True)
print(f"AUC = {roc_auc:.4f}")

# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic – Test set")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()

print("Saving ROC curve figure...", flush=True)
plt.savefig(OUTPUT_FIGURE, dpi=300)
print(f"Saved ROC curve to {OUTPUT_FIGURE}", flush=True) 