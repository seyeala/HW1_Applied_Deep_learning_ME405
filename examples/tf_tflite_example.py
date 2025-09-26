
# examples/tf_tflite_example.py
import os, numpy as np, tensorflow as tf
from pathlib import Path
from PIL import Image
from mobile_gradio_classifier import MobileClassifierApp

TFLITE_PATH = os.environ.get("TFLITE_PATH", "model.tflite")
IMG_SIZE = int(os.environ.get("IMG_SIZE", "224"))
DEFAULT_CLASSES = ["class_one", "class_two"]
classes_path = Path(__file__).with_name("classes.txt")
if classes_path.exists():
    classes = [line.strip() for line in classes_path.read_text().splitlines() if line.strip()] or DEFAULT_CLASSES
else:
    classes = DEFAULT_CLASSES

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
in_dtype = inp["dtype"]

def _prep(pil: Image.Image):
    arr = tf.image.resize(np.array(pil.convert("RGB")), (IMG_SIZE, IMG_SIZE)).numpy()
    if in_dtype == np.uint8: arr = arr.astype(np.uint8)
    else: arr = (arr.astype("float32")/255.0).astype(in_dtype)
    return arr[None, ...]

def predict_fn(pil: Image.Image):
    x = _prep(pil)
    interpreter.set_tensor(inp["index"], x)
    interpreter.invoke()
    y = interpreter.get_tensor(out["index"])[0]
    if (y < 0).any() or not np.allclose(y.sum(), 1.0, atol=1e-3):
        y = tf.nn.softmax(y).numpy()
    return {c: float(p) for c,p in zip(classes, y)}

app = MobileClassifierApp(classes, predict_fn=predict_fn)
app.launch(share=True)
