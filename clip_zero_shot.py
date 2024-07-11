import os
import torch
from PIL import Image
import open_clip
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision.datasets import CIFAR100, CIFAR10

image_name= "shadow_of_cats"
image_path = f"source_image/{image_name}.jpeg"  # Change this to your local image path

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-B-32')

image = preprocess(Image.open(image_path)).unsqueeze(0)

cifar10 = CIFAR10(os.path.expanduser("~/.cache"), transform=preprocess, download=True)
text_descriptions = [f"A photo of a {label}" for label in cifar10.classes]

cifar100 = CIFAR100(os.path.expanduser("~/.cache"), transform=preprocess, download=True)
text_descriptions_100 = [f"A photo of a {label}" for label in cifar100.classes]

text_descriptions.extend(text_descriptions_100)
print(text_descriptions)
text_tokens = tokenizer(text_descriptions)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_tokens)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)

# print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

# Sample data
probabilities = text_probs[0]

# Load the image
image = Image.open(image_path)

# Create the figure and the axes
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Display the image on the left
ax[0].imshow(image)
ax[0].axis('off')  # Hide the axes

# Create the bar plot on the right
ax[1].barh([text_descriptions[index] for index in top_labels[0].numpy()], top_probs[0], color='skyblue')
# Farben für die y-Achsenbeschriftungen
colors = ['green', 'red', 'red', 'red', 'red']
# Y-Achsenbeschriftungen manuell ändern
for ytick, color in zip(ax[1].get_yticklabels(), colors):
    ytick.set_color(color)
ax[1].set_xlim(0, 1)
ax[1].set_xlabel('Probability')
ax[1].set_title('Predicted Probabilities')

# Adjust layout to avoid overlap
plt.tight_layout()

# Save the figure
plt.savefig(f"target_images/zero_shot_{image_name}.png", bbox_inches='tight')