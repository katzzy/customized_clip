import torch
from PIL import Image
from model.clip.clip_model import load_clip_model
from model.clip.clip_transform import load_clip_transform
from model.clip.clip_tokenize import SimpleTokenizer
from model.clip.clip_tokenize import tokenize

device = "cuda" if torch.cuda.is_available() else "cpu"

model_config_path = "model/model_configs/ViT-L-14-336px.yaml"
model = load_clip_model(model_config_path).to(device).eval()

preprocess = load_clip_transform(model.visual.input_resolution)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)

_tokenizer = SimpleTokenizer()
text = tokenize(_tokenizer, ["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9325925  0.06268822 0.00471936]]
