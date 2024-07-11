import torch
from PIL import Image
import open_clip
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

image_name= "stair_cat"
image_path = f"source_image/{image_name}.jpeg"  # Change this to your local image path
if image_name == "butty":
    labels = ["A photo of an orange cat, which lays on its back on a wooden floor.", "A photo of an orange dog, which lays on its back on a wooden floor.",  "A photo of a wooden floor, which lays on its back on an orange cat.", "Of an cat wooden on photo, which on lays orange its a back floor."]
    label = ["A photo of an orange cat, which\nlays on its back on a wooden floor.", "A photo of an orange dog, which\nlays on its back on a wooden floor.",  "A photo of a wooden floor, which\nlays on its back on an orange cat.", "Of an cat of wooden photo, which\non lays orange its a back floor."]
elif image_name == "butty_in_a_bag":
    labels = ["A photo of an orange cat, which sits in a plastic bag.", "A photo of an orange dog, which sits in a plastic bag.",  "A photo of a plastic bag, which sits in an orange cat.",  "A plastic bag sits in an orange cat, which a photo of."]
    label = ["A photo of an orange cat, which\nsits in a plastic bag.", "A photo of an orange dog, which\nsits in a plastic bag.",  "A photo of a plastic bag, which\nsits in an orange cat.", "A plastic bag sits in an orange\ncat, which a photo of."]
elif image_name == "shadow_of_cats":
    labels = ["A photo of the shadow of two cats with the sky in the background.", "A photo of the shadow of two cats with superman flying in the background.",  "A photo of the shadow of two sky with the cat in the background.",  "Background with shadow the cats two of photo the in sky a."]
    label = ["A photo of the shadow of two cats\nwith the sky in the background.", "A photo of the shadow of two cats\nwith superman flying in the background.",  "A photo of the shadow of two sky\nwith the cat in the background.", "Background with shadow the cats\ntwo of photo the in sky a."]
elif image_name == "stair_cat":
    labels = ["A photo of a cat hidden behind the stairs.", "A photo of a cat hidden behind elephants.",  "A photo of a stairs hidden behind cat.",  "Hidden behind a cat, a photo of stairs."]
    label = ["A photo of a cat hidden\nbehind the stairs.", "A photo of a cat hidden\nbehind elephants.",  "A photo of a stairs hidden\nbehind cat.", "Hidden behind a cat, a\nphoto of stairs."]

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-B-32')

image = preprocess(Image.open(image_path)).unsqueeze(0)
text = tokenizer(labels)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).tolist()

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

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
ax[1].barh(label, probabilities, color='skyblue')
# Farben für die y-Achsenbeschriftungen
colors = ['green', 'red', 'red', 'red']
# Y-Achsenbeschriftungen manuell ändern
for ytick, color in zip(ax[1].get_yticklabels(), colors):
    ytick.set_color(color)
ax[1].set_xlim(0, 1)
ax[1].set_xlabel('Probability')
ax[1].set_title('Predicted Probabilities')

# Adjust layout to avoid overlap
plt.tight_layout()

# Save the figure
plt.savefig(f"target_images/bow_{image_name}.png", bbox_inches='tight')