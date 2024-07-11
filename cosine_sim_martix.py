import os
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from collections import OrderedDict
import torch
import open_clip
from open_clip import tokenizer
import matplotlib 


model, _, preprocess = open_clip.create_model_and_transforms('convnext_base_w', pretrained='laion2b_s13b_b82k_augreg')

# images in skimage to use and their textual descriptions
descriptions = {
    "butty": "A photo of an orange cat, which\nlays on its back on a wooden floor.",
    "butty_in_a_bag": "A photo of an orange cat, which\nsits in a plastic bag.",
    "shadow_of_cats": "A photo of the shadow of two\ncats with the sky in the background.",
    "stair_cat": "A photo of a cat hidden behind\nthe stairs."
}

original_images = []
images = []
texts = []

for filename in descriptions.keys():
    name = os.path.splitext(filename)[0]
    if name not in descriptions:
        continue

    image = Image.open(f"source_image/{filename}.jpeg").convert("RGB")

    original_images.append(image)
    images.append(preprocess(image))
    texts.append(descriptions[name])

image_input = torch.tensor(np.stack(images))
text_tokens = tokenizer.tokenize(["This is " + desc for desc in texts])


with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    text_features = model.encode_text(text_tokens).float()


image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T


count = len(descriptions)

plt.figure(figsize=(10, 5))
plt.imshow(similarity, vmin=np.min(similarity), vmax=np.max(similarity))
# plt.colorbar()
plt.yticks(range(count), texts, fontsize=14)
plt.xticks([])
for i, image in enumerate(original_images):
    plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
for x in range(similarity.shape[1]):
    for y in range(similarity.shape[0]):
        plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

for side in ["left", "top", "right", "bottom"]:
  plt.gca().spines[side].set_visible(False)

plt.xlim([-0.5, count - 0.5])
plt.ylim([count + 0.5, -2])


# Adjust layout to avoid overlap
plt.tight_layout()

# Save the figure
plt.savefig(f"target_images/cos_sin_matrix.png", bbox_inches='tight')