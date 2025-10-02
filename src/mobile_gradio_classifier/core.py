
"""
mobile_gradio_classifier.py
---------------------------
A model-agnostic, resolution-agnostic Gradio app helper for image *and* video
classification, designed for quick deployment from Colab and usable on phones.

Key features
- Plug in *any* classifier via a `predict_fn(PIL.Image) -> Dict[label, prob]`
  OR pass a PyTorch `nn.Module` with a `preprocess(image: PIL.Image) -> Tensor`.
- Accepts images from upload or webcam; accepts videos (upload or webcam).
- For video, samples frames at a target FPS, runs classification, aggregates
  predictions (majority vote, average probs), and optionally sends an email
  when a target class is detected with confidence ≥ threshold.
- Minimal dependencies: gradio, pillow, numpy; uses OpenCV if present, else
  falls back to imageio for video frame extraction.
- Phone friendly: two tabs (Image / Video), clear outputs, and a single
  `launch(share=True)` for quick testing on mobile.
"""

from __future__ import annotations
import os
import io
import time
import smtplib
import ssl
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import gradio as gr

from .video_utils import iter_video_frames


@dataclass
class EmailConfig:
    smtp_host: str
    smtp_port: int = 587
    username: Optional[str] = None
    password: Optional[str] = None  # prefer app password
    use_tls: bool = True
    from_addr: Optional[str] = None
    to_addrs: List[str] = field(default_factory=list)
    subject: str = "Classifier Alert"
    # If provided, only send if *any* of these labels are seen
    trigger_labels: Optional[List[str]] = None
    # Confidence threshold for sending an alert
    min_confidence: float = 0.9


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class MobileClassifierApp:
    """
    Model-agnostic Gradio wrapper.
    Use either `predict_fn` OR (`torch_model` + `preprocess_fn`).
    """

    def __init__(
        self,
        classes: List[str],
        predict_fn: Optional[Callable[[Image.Image], Dict[str, float]]] = None,
        torch_model: Optional["torch.nn.Module"] = None,
        preprocess_fn: Optional[Callable[[Image.Image], "torch.Tensor"]] = None,
        device: Optional[str] = None,
        email_config: Optional[EmailConfig] = None,
        default_video_fps: float = 2.0,
    ) -> None:
        self.classes = classes
        self.predict_fn = predict_fn
        self.model = torch_model
        self.preprocess_fn = preprocess_fn
        self.email = email_config
        self.default_video_fps = default_video_fps

        if self.predict_fn is None and (self.model is None or self.preprocess_fn is None):
            raise ValueError("Provide either predict_fn OR (torch_model and preprocess_fn).")

        # Torch is optional at import time; only needed if a model was passed
        if self.model is not None:
            import torch  # local import
            self.torch = torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device).eval()
        else:
            self.torch = None
            self.device = None

    # ---------- Core Prediction ----------
    def _predict_image(self, pil_img: Image.Image) -> Dict[str, float]:
        """Return a dict {label: prob} for a single image."""
        if self.predict_fn is not None:
            out = self.predict_fn(pil_img)
            # normalize if needed
            probs = np.array([out.get(c, 0.0) for c in self.classes], dtype=float)
            if not np.allclose(probs.sum(), 1.0, atol=1e-3):
                # try softmax if outputs look like logits
                probs = _softmax(probs)
            return {c: float(p) for c, p in zip(self.classes, probs)}

        # Torch path
        assert self.model is not None and self.preprocess_fn is not None and self.torch is not None
        with self.torch.no_grad():
            x = self.preprocess_fn(pil_img).unsqueeze(0).to(self.device)
            logits = self.model(x)
            if logits.ndim == 2 and logits.shape[1] == len(self.classes):
                probs = self.torch.softmax(logits, dim=1)[0].cpu().numpy()
            else:
                # fall back: try softmax on raw outputs
                probs = _softmax(logits.flatten().cpu().numpy())
            return {c: float(p) for c, p in zip(self.classes, probs)}

    # ---------- Email ----------
    def _send_email_if_triggered(self, label: str, prob: float, extra: str = "") -> str:
        if not self.email:
            return "Email disabled."
        if self.email.trigger_labels and label not in self.email.trigger_labels:
            return f"No email: label '{label}' not in trigger list."
        if prob < self.email.min_confidence:
            return f"No email: confidence {prob:.2f} < threshold {self.email.min_confidence:.2f}."
        try:
            msg_lines = [
                f"Subject: {self.email.subject}",
                f"From: {self.email.from_addr or self.email.username}",
                f"To: {', '.join(self.email.to_addrs)}",
                "",
                f"Detected: {label} (p={prob:.3f})",
            ]
            if extra:
                msg_lines.append(extra)
            msg = "\r\n".join(msg_lines).encode("utf-8")

            context = ssl.create_default_context()
            if self.email.use_tls:
                with smtplib.SMTP(self.email.smtp_host, self.email.smtp_port) as server:
                    server.starttls(context=context)
                    if self.email.username and self.email.password:
                        server.login(self.email.username, self.email.password)
                    server.sendmail(self.email.from_addr or self.email.username, self.email.to_addrs, msg)
            else:
                with smtplib.SMTP(self.email.smtp_host, self.email.smtp_port) as server:
                    if self.email.username and self.email.password:
                        server.login(self.email.username, self.email.password)
                    server.sendmail(self.email.from_addr or self.email.username, self.email.to_addrs, msg)
            return "Email sent."
        except Exception as e:
            return f"Email error: {e}"

    # ---------- Gradio Handlers ----------
    def predict_image_gr(self, img: np.ndarray, send_email: bool = False) -> Tuple[str, Dict[str, float], str]:
        if img is None:
            return "", {}, "No image provided."
        pil = Image.fromarray(img).convert("RGB")
        probs = self._predict_image(pil)
        top_label = max(probs, key=probs.get)
        msg = ""
        if send_email:
            msg = self._send_email_if_triggered(top_label, probs[top_label])
        return top_label, probs, msg

    def predict_video_gr(
        self,
        video_path: str,
        target_fps: float,
        aggregate: str = "majority",
        send_email: bool = False,
    ) -> Tuple[str, str]:
        if not video_path:
            return "", "No video provided."
        # Process frames
        labels = []
        probs_list = []
        t0 = time.time()
        n_frames = 0
        for frame in iter_video_frames(video_path, target_fps=target_fps or self.default_video_fps):
            pr = self._predict_image(frame)
            probs_list.append([pr[c] for c in self.classes])
            labels.append(max(pr, key=pr.get))
            n_frames += 1

        if n_frames == 0:
            return "", "No frames processed."

        probs_arr = np.array(probs_list)  # shape [N, C]
        if aggregate == "avg":
            mean_probs = probs_arr.mean(axis=0)
            top_idx = int(mean_probs.argmax())
        else:
            # majority vote
            counts = {c: labels.count(c) for c in self.classes}
            top_idx = int(np.argmax([counts[c] for c in self.classes]))

        top_label = self.classes[top_idx]
        conf = float(probs_arr.mean(axis=0)[top_idx])  # report avg conf of top
        msg = f"Frames processed: {n_frames}. Top result: {top_label} (avg p={conf:.3f})."
        if send_email:
            msg += " " + self._send_email_if_triggered(top_label, conf, extra=f"[video] {os.path.basename(video_path)}")
        return top_label, msg

    def _blank_live_state(self) -> Dict[str, object]:
        return {"last_time": 0.0, "label": "", "probs": {}}

    def reset_live_state(self, _: bool) -> Tuple[str, Dict[str, float], Dict[str, object]]:
        state = self._blank_live_state()
        return "", {}, state

    def predict_live_gr(
        self,
        frame: Optional[np.ndarray],
        freq_hz: float,
        active: bool,
        state: Optional[Dict[str, object]],
    ) -> Tuple[str, Dict[str, float], Dict[str, object]]:
        state = state or self._blank_live_state()
        if not active or frame is None:
            return state.get("label", ""), state.get("probs", {}), state

        min_interval = 1.0 / max(freq_hz, 1e-6)
        last_time = float(state.get("last_time", 0.0))
        now = time.time()

        if now - last_time < min_interval and state.get("label"):
            return state["label"], state["probs"], state

        pil = Image.fromarray(frame).convert("RGB")
        probs = self._predict_image(pil)
        top_label = max(probs, key=probs.get)

        new_state = {"last_time": now, "label": top_label, "probs": probs}
        return top_label, probs, new_state

    # ---------- Build UI ----------
    def build_demo(self) -> gr.Blocks:
        with gr.Blocks(title="Mobile Classifier", css="footer {visibility: hidden}") as demo:
            gr.Markdown("## 📱 Mobile Classifier (Image & Video)\nWorks with any classifier.")

            with gr.Tab("Image"):
                with gr.Row():
                    img_in = gr.Image(sources=["upload", "webcam"], label="Upload or take a photo", streaming=False)
                    with gr.Column():
                        send_email_chk = gr.Checkbox(label="Send email on detection", value=False)
                        out_label = gr.Label(label="Top Prediction")
                        out_probs = gr.Label(label="Class Probabilities")
                        out_msg = gr.Textbox(label="Log", interactive=False)

                img_btn = gr.Button("Predict")
                img_btn.click(self.predict_image_gr, inputs=[img_in, send_email_chk], outputs=[out_label, out_probs, out_msg])

            with gr.Tab("Video"):
                with gr.Row():
                    vid_in = gr.Video(
                        label="Upload or record a short video",
                        sources=["upload", "webcam"],
                    )
                with gr.Row():
                    fps_in = gr.Slider(0.5, 10.0, value=self.default_video_fps, step=0.5, label="Target FPS for sampling")
                    agg_in = gr.Dropdown(choices=["majority", "avg"], value="majority", label="Aggregation")
                    send_email_chk2 = gr.Checkbox(label="Send email on detection", value=False)
                with gr.Row():
                    vid_out_label = gr.Textbox(label="Video Result", interactive=False)
                    vid_log = gr.Textbox(label="Log", interactive=False, lines=2)

                vid_btn = gr.Button("Run on Video")
                vid_btn.click(self.predict_video_gr, inputs=[vid_in, fps_in, agg_in, send_email_chk2], outputs=[vid_out_label, vid_log])

            with gr.Tab("Live Video"):
                gr.Markdown("### Continuous webcam monitoring\nToggle on to classify frames automatically at a fixed frequency.")
                with gr.Row():
                    live_toggle = gr.Checkbox(label="Enable live classification", value=False)
                    live_freq = gr.Slider(0.5, 5.0, value=1.0, step=0.5, label="Classification frequency (Hz)")
                with gr.Row():
                    live_feed = gr.Image(label="Webcam stream", sources=["webcam"], streaming=True)
                    with gr.Column():
                        live_label = gr.Label(label="Live Top Prediction")
                        live_probs = gr.Label(label="Live Class Probabilities")
                live_state = gr.State(self._blank_live_state())
                live_feed.stream(
                    self.predict_live_gr,
                    inputs=[live_feed, live_freq, live_toggle, live_state],
                    outputs=[live_label, live_probs, live_state],
                )
                live_toggle.change(
                    self.reset_live_state,
                    inputs=[live_toggle],
                    outputs=[live_label, live_probs, live_state],
                )

            gr.Markdown("Tip: Click 'Share' in `launch()` to test from your phone.")
        return demo

    def launch(self, **kwargs):
        demo = self.build_demo()
        return demo.launch(**kwargs)
