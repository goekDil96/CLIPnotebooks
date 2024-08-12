import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import open_clip
import torch

import matplotlib 
plt.rc('font', family='serif',size=24)
matplotlib.rc('text', usetex=True)
matplotlib.rc('legend', fontsize=24)
matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'

# Load the image
image_name = "sketch_cat"
image_path = f'source_image/{image_name}.png'  # Change this to the path of your image
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-B-32')

labels = ["noise image", "cheetah", "dog", "lion", "cat"]

prob_noise = []
prob_cheet = []
prob_dog = []
prob_lion = []
prob_cat = []

def clip_probs(image, labels=labels):
    text = tokenizer(labels)
    image = preprocess(Image.open(image)).unsqueeze(0)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).tolist()

    # Sample data
    probabilities = text_probs[0]
    return probabilities

##
stds = [50, 100, 150, 200, 250]

# Create the figure and the axes
fig, ax = plt.subplots(len(stds), 4, figsize=(22, 5*len(stds)))

for i, std in enumerate(stds):
    for c in [0, 25]:
        noise = np.random.normal(0, std + c, gray_image.shape)
        noisy_image = gray_image + noise

        output_path = f'noisy/noisy_image_std_{std + c}.jpeg'
        cv2.imwrite(output_path, noisy_image)

        # Display the noisy image on the left
        ax[i, 0 if c == 0 else 2].imshow(noisy_image, cmap='gray', vmin=0, vmax=255)
        ax[i, 0 if c == 0 else 2].axis('off')  # Hide the axes
        ax[i, 0 if c == 0 else 2].set_title(f'Noisy Image with std={std + c}')

        probs = clip_probs(output_path)
        # Create the dummy bar plot on the right
        ax[i, 1 if c == 0 else 3].barh(labels, probs, color='skyblue')
        colors = ['red', 'red', 'red', 'red', 'green']
        # Y-Achsenbeschriftungen manuell ändern
        for ytick, color in zip(ax[i, 1 if c == 0 else 3].get_yticklabels(), colors):
            ytick.set_color(color)
        ax[i, 1 if c == 0 else 3].set_xlim(0, 1)
        ax[i, 1 if c == 0 else 3].set_xlabel('Probability')
        ax[i, 1 if c == 0 else 3].set_title('Predicted Probabilities')

        prob_noise.append(probs[0])
        prob_cheet.append(probs[1])
        prob_dog.append(probs[2])
        prob_lion.append(probs[3])
        prob_cat.append(probs[4])

plt.tight_layout()

# Save the figure
plt.savefig(f"target_images/gauss_{image_name}.png", bbox_inches='tight')


# Create a figure and axis
fig, ax = plt.subplots(figsize=(24,12))

stds = [i for i in range(25, 275, 25)]
# Plot the lines
ax.scatter(stds, prob_cat, label='cat', marker="x", color="red")
ax.scatter(stds, prob_noise, label='noise image', marker="x", color="blue")
ax.scatter(stds, prob_cheet, label='cheetah', marker="x", color="green")
ax.scatter(stds, prob_dog, label='dog', marker="x", color="orange")
ax.scatter(stds, prob_lion, label='lion', marker="x", color="purple")

ax.plot(stds, prob_cat, color="red")
ax.plot(stds, prob_noise, color="blue")
ax.plot(stds, prob_cheet, color="green")
ax.plot(stds, prob_dog, color="orange")
ax.plot(stds, prob_lion, color="purple")


# Set labels and title
ax.set_xlabel('std')
ax.set_ylabel('Probability')
# Display legend
ax.legend()

# Adjust layout to avoid overlap
plt.tight_layout()

# Save the figure
plt.savefig(f"target_images/gauss_line_{image_name}.png", bbox_inches='tight')