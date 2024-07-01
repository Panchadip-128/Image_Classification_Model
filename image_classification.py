import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image


feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')


image_path = 'example_image.jpg'
image = Image.open(image_path)
inputs = feature_extractor(images=image, return_tensors="pt")


outputs = model(**inputs)
logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print(f"Predicted class index: {predicted_class_idx}")
