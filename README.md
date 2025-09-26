# HW1_Applied_Deep_learning_ME405

# mobile-classifier-repo

A **model-agnostic, resolution-agnostic** Gradio wrapper for **image and video classification**, built to be **pippable** and easy to run on phones via `share=True` in Colab.

## Features
- 📷 Image input (upload / webcam) & 🎥 Video input (upload / camera)
- ⏱ Sample video at target FPS; aggregate by **majority** or **average prob**
- 🔌 Works with **any classifier**:
  - Provide a `predict_fn(PIL.Image)->Dict[label, prob]`, **or**
  - Plug a **PyTorch** model + `preprocess_fn`, or use **TensorFlow/TFLite** with a `predict_fn`
- ✉️ Optional email alerts when a label is detected above threshold
- 📱 One line to launch a public link for mobile testing

## Install (editable dev mode)
```bash
git clone <your-repo-url>.git
cd mobile-classifier-repo
pip install -e .[video]        # add [torch], [tf], [tflite] as needed
```

## Quick usage (Torch example)
```python
from mobile_gradio_classifier import MobileClassifierApp

# define classes
classes = ["occupied", "empty"]

# Torch route:
import torch
from torchvision import models, transforms
from PIL import Image

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("final.pt", map_location="cpu"))
model.eval()

preprocess = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
def preprocess_fn(pil: Image.Image):
    return preprocess(pil.convert("RGB"))

app = MobileClassifierApp(classes, torch_model=model, preprocess_fn=preprocess_fn)
app.launch(share=True)
```

## Quick usage (framework-agnostic via `predict_fn`)
```python
from mobile_gradio_classifier import MobileClassifierApp
from PIL import Image

classes = ["on", "off"]

def predict_fn(pil: Image.Image):
    # run your TF/ONNX/custom model here, return {label: prob}
    return {"on": 0.12, "off": 0.88}

app = MobileClassifierApp(classes, predict_fn=predict_fn)
app.launch(share=True)
```

## Examples
- `examples/torch_example.py` — PyTorch ResNet-18 head
- `examples/tf_keras_example.py` — Keras/SavedModel
- `examples/tf_tflite_example.py` — TFLite interpreter

All three scripts look for a `classes.txt` file that lives right next to them in `examples/` (a starter copy with two placeholder labels ships with the repo).
Edit that file with one label per line to match your model's outputs. If you delete it or leave it empty, the scripts will fall back to a pair of dummy labels so you can still launch the UI for smoke testing.

## Module API

### `MobileClassifierApp`
```python
MobileClassifierApp(
    classes: list[str],
    predict_fn: Optional[Callable[[PIL.Image], dict[str, float]]] = None,
    torch_model: Optional[torch.nn.Module] = None,
    preprocess_fn: Optional[Callable[[PIL.Image], "torch.Tensor"]] = None,
    device: Optional[str] = None,
    email_config: Optional[EmailConfig] = None,
    default_video_fps: float = 2.0,
)
```
- **Use either** `predict_fn` **or** `(torch_model + preprocess_fn)`.
- `.build_demo()` -> `gr.Blocks`
- `.launch(**kwargs)` -> runs Gradio app

### `EmailConfig`
Fields:
- `smtp_host`, `smtp_port`, `username`, `password`, `use_tls`
- `from_addr`, `to_addrs`, `subject`
- `trigger_labels` (list[str] | None), `min_confidence` (float)

## Repo layout
```
mobile-classifier-repo/
├─ src/
│  └─ mobile_gradio_classifier/
│     ├─ __init__.py
│     └─ core.py
├─ examples/
│  ├─ torch_example.py
│  ├─ tf_keras_example.py
│  └─ tf_tflite_example.py
├─ hw/
│  ├─ assignment.tex
│  └─ LICENSE-CC-BY-NC-SA.txt
├─ notebooks/
│  └─ demo_colab_stub.ipynb
├─ docs/
│  └─ API.md
├─ README.md
├─ pyproject.toml
├─ LICENSE  (MIT for code)
└─ .gitignore
```

## License
- Code: **MIT** (see `LICENSE`)
- Homework text / LaTeX: **CC BY-NC-SA 4.0** (see `hw/LICENSE-CC-BY-NC-SA.txt`)
