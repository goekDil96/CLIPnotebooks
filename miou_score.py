import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import matplotlib 
plt.rc('font', family='serif',size=16)
matplotlib.rc('text', usetex=True)
matplotlib.rc('legend', fontsize=16)
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{xcolor}'


# Define the colormap
cmap = LinearSegmentedColormap.from_list('rg', ['red', 'white', 'green'], N=256)

def compute_iou_for_class(pred_mask, gt_mask, target_class):
    """
    Compute Intersection over Union (IoU) for a specific class.
    
    Parameters:
    - pred_mask: numpy array of shape (H, W) with predicted segmentation
    - gt_mask: numpy array of shape (H, W) with ground truth segmentation
    - target_class: integer, the class for which IoU needs to be computed
    
    Returns:
    - iou: float, Intersection over Union score for the target class
    """
    pred_cls = (pred_mask == target_class)
    gt_cls = (gt_mask == target_class)
    
    intersection = np.logical_and(pred_cls, gt_cls).sum()
    union = np.logical_or(pred_cls, gt_cls).sum()
    
    if union == 0:
        iou = float('nan')  # Avoid division by zero
    else:
        iou = intersection / union
    
    return iou

def plot_masks(pred_mask, gt_mask, target_class):
    """
    Plot the predicted and ground truth masks, highlighting correct and incorrect pixels for a specific class.
    
    Parameters:
    - pred_mask: numpy array of shape (H, W) with predicted segmentation
    - gt_mask: numpy array of shape (H, W) with ground truth segmentation
    - target_class: integer, the class for which pixels are highlighted
    """
    plt.figure(figsize=(12, 6))
    
    # Plot Predicted Mask
    plt.subplot(1, 4, 2)
    plt.imshow(pred_mask, cmap='binary', vmin=0, vmax=np.max(pred_mask))
    plt.xticks([])
    plt.yticks([])
    plt.title('Predicted Mask')
    
    # Plot Ground Truth Mask
    plt.subplot(1, 4, 1)
    plt.imshow(gt_mask, cmap='binary', vmin=0, vmax=np.max(gt_mask))
    plt.xticks([])
    plt.yticks([])
    plt.text(0.5, 3.5, r'Class 0: $\square$, Class 1: $\blacksquare$', fontsize=14, 
             ha="center", va="center",
         bbox=dict(boxstyle="round",
                   ec=(0., 0., 0.),
                   fc=(1., 1., 1.),
                   ))
    plt.title('Ground Truth Mask')
    
    # Highlight Correct and Incorrect Pixels for the target class
    pred_cls = (pred_mask == gt_mask) 
    gt_cls = (gt_mask != 0) & (pred_mask != 0)
    
    combined_mask = np.zeros_like(pred_mask)
    combined_mask[pred_cls] = 2  # Correct predictions
    combined_mask[gt_cls] = 1  # Incorrect predictions

    plt.subplot(1, 4, 3)
    plt.imshow(combined_mask, cmap=cmap, vmin=0, vmax=2)
    plt.xticks([])
    plt.yticks([])
    iou = compute_iou_for_class(pred_mask, gt_mask, 0)
    plt.text(2.5, 0, f'IoU: {iou:.2f}', fontsize=14, ha="center", va="center",
         bbox=dict(boxstyle="round",
                   ec=(0., 0., 0.),
                   fc=(1., 1., 1.),
                   ))

    plt.title('IoU Class 0')
    
    # Highlight Correct and Incorrect Pixels for the target class
    pred_cls = (pred_mask == gt_mask) 
    gt_cls = (gt_mask != 1) & (pred_mask != 1)
    
    combined_mask = np.zeros_like(pred_mask)
    combined_mask[pred_cls] = 2  # Correct predictions
    combined_mask[gt_cls] = 1  # Incorrect prediction
    plt.subplot(1, 4, 4)
    plt.imshow(combined_mask, cmap=cmap, vmin=0, vmax=2)
    plt.xticks([])
    plt.yticks([])
    iou = compute_iou_for_class(pred_mask, gt_mask, 1)
    plt.text(2.5, 0, f'IoU: {iou:.2f}', fontsize=14, ha="center", va="center",
         bbox=dict(boxstyle="round",
                   ec=(0., 0., 0.),
                   fc=(1., 1., 1.),
                   ))

    plt.title('IoU Class 1')

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Example segmentation masks
    pred_mask = np.array([
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ])

    gt_mask = np.array([
        [0, 1, 1, 1],
        [0, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 0, 1, 1]
    ])

    target_class = 1  # Compute and plot IoU for class 1

    iou = compute_iou_for_class(pred_mask, gt_mask, target_class)
    print(f"IoU for class {target_class}: {iou:.4f}")
    
    # Plot the masks
    plot_masks(pred_mask, gt_mask, target_class)
