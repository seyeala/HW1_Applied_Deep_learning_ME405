
# examples/torch_example.py
from mobile_gradio_classifier import MobileClassifierApp
import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

DEFAULT_CLASSES = ["class_one", "class_two"]
classes_path = Path(__file__).with_name("classes.txt")
if classes_path.exists():
    classes = [line.strip() for line in classes_path.read_text().splitlines() if line.strip()] or DEFAULT_CLASSES
else:
    classes = DEFAULT_CLASSES

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
device = "cuda" if torch.cuda.is_available() else "cpu"
model.load_state_dict(torch.load("final.pt", map_location=device))
model.to(device).eval()

preprocess = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
def preprocess_fn(pil: Image.Image):
    return preprocess(pil.convert("RGB"))

app = MobileClassifierApp(classes, torch_model=model, preprocess_fn=preprocess_fn)
app.launch(share=True)
