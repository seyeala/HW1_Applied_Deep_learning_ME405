# HW1_Applied_Deep_learning_ME405

# mobile-classifier-repo

A **model-agnostic, resolution-agnostic** Gradio wrapper for **image and video classification**, built to be **pippable** and easy to run on phones via `share=True` in Colab.

## Features
- 📷 Image input (upload by default; set `source="webcam"` in `core.py` to capture directly) & 🎥 Video input (upload by default; set `source="webcam"` to record)
- ⏱ Sample video at target FPS; aggregate by **majority** or **average prob**
- 🔁 Optional continuous webcam mode with adjustable classification frequency
- 🔌 Works with **any classifier**:
  - Provide a `predict_fn(PIL.Image)->Dict[label, prob]`, **or**
  - Plug a **PyTorch** model + `preprocess_fn`, or use **TensorFlow/TFLite** with a `predict_fn`
- ✉️ Optional email alerts when a label is detected above threshold
- 📱 One line to launch a public link for mobile testing

> **Gradio compatibility:** The project now pins **Gradio 3.50.2** for long-term stability. This release expects a single `source` string per media component (e.g., `"upload"` or `"webcam"`). The defaults favor uploads, but you can switch sources in `MobileClassifierApp.build_demo()` if you want webcam-first behavior.

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

To plug in a real TensorFlow/Keras model for the `tf_keras_example.py` demo, export it as a `.keras` file and point the script to it with the `MODEL_PATH` environment variable:

```bash
MODEL_PATH=/path/to/your/model.keras python examples/tf_keras_example.py
```
If that variable is unset or the file cannot be found, the example will build a lightweight in-memory demo network so you can still try the app flow without a trained model.

## Exporting frames from videos

Use the `mobile_gradio_classifier.export_frames` module (or the `mobile-export-frames` console entry point) to turn a batch of videos into resized image sequences that mirror the structure expected by `MobileClassifierApp`. A sample configuration lives at [`examples/frame_export.yaml`](examples/frame_export.yaml).

```bash
python -m mobile_gradio_classifier.export_frames --config examples/frame_export.yaml
# or
mobile-export-frames --config examples/frame_export.yaml
```

The tool accepts one or more `input_glob` patterns and writes every matched video to its own subdirectory inside `output_dir`. For example, when `output_dir` is `exports/frames` and a source file named `clip.mp4` is processed, frames are saved under `exports/frames/clip/clip_000000.png`, `clip_000001.png`, and so on. File names include a zero-padded index that reflects the sampling order at the requested `fps`, and the image format is derived from the `format` value in the config (`png`, `jpeg`, etc.). Frames are resized to `size` using Pillow's high-quality `LANCZOS` filter.

### Optional dependencies

Reading videos requires either `opencv-python` or `imageio`; the exporter will automatically prefer OpenCV and fall back to ImageIO if only that package is available. Install whichever backend fits your environment:

```bash
pip install mobile-gradio-classifier[video]
# or explicitly
pip install opencv-python  # OpenCV backend
pip install imageio        # ImageIO fallback backend
```

Resizing is handled by Pillow (installed with the base package). Its `LANCZOS` filter ensures high-quality downsampling, which is particularly helpful when generating datasets for model training or validation.

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
