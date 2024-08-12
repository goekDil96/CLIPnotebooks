import matplotlib.pyplot as plt
import matplotlib.patches as patches

import matplotlib 
plt.rc('font', family='serif',size=24)
matplotlib.rc('text', usetex=True)
matplotlib.rc('legend', fontsize=24)
matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'


def iou(boxA, boxB):
    # boxA and boxB are both in the form [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight
    
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou, (xA, yA, xB, yB)

def plot_boxes(boxA, boxB, intersection):
    fig, ax = plt.subplots(1)
    
    # Create a Rectangle patch for each box
    rectA = patches.Rectangle((boxA[0], boxA[1]), boxA[2]-boxA[0], boxA[3]-boxA[1], linewidth=1, edgecolor='r', facecolor='none', label='Box A')
    rectB = patches.Rectangle((boxB[0], boxB[1]), boxB[2]-boxB[0], boxB[3]-boxB[1], linewidth=1, edgecolor='b', facecolor='none', label='Box B')
    
    # Create a Rectangle patch for the intersection
    if intersection[2] > intersection[0] and intersection[3] > intersection[1]:
        rectIntersection = patches.Rectangle((intersection[0], intersection[1]), intersection[2]-intersection[0], intersection[3]-intersection[1], linewidth=1, edgecolor='g', facecolor='none', label='Intersection')
        ax.add_patch(rectIntersection)
    
    # Add the patches to the Axes
    ax.add_patch(rectA)
    ax.add_patch(rectB)
    
    # Set the limits of the plot
    ax.set_xlim(0, max(boxA[2], boxB[2]) + 1)
    ax.set_ylim(0, max(boxA[3], boxB[3]) + 1)
    
    # Add legend
    ax.legend()
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Example usage
boxA = [1, 1, 3, 3]
boxB = [2, 2, 4, 4]

iou_score, intersection = iou(boxA, boxB)
print(f"The IoU score is: {iou_score}")

plot_boxes(boxA, boxB, intersection)
