# 0. Hugging Face ZeroGPU compatibility: MUST be imported before torch or any CUDA library!
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(fn=None, **kwargs):
            if fn is None:
                return lambda f: f
            return fn

import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageDraw, ImageFont


# ============================================================================
# 1. Model Architecture
# ============================================================================

class Block(nn.Module):
    """Two 3x3 convolutions with GroupNorm and residual projection."""
    def __init__(self, cin: int, cout: int, tdim: int):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.norm = nn.GroupNorm(4, cout)
        self.t = nn.Linear(tdim, cout)
        self.proj = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x: torch.Tensor, te: torch.Tensor) -> torch.Tensor:
        h = self.norm(self.c1(x)) + self.t(te)[:, :, None, None]
        change = self.c2(F.silu(h))
        return self.proj(x) + change


class Level(nn.Module):
    """Two blocks in a row; the first changes the channel count."""
    def __init__(self, cin: int, cout: int, tdim: int):
        super().__init__()
        self.b1 = Block(cin, cout, tdim)
        self.b2 = Block(cout, cout, tdim)

    def forward(self, x: torch.Tensor, te: torch.Tensor) -> torch.Tensor:
        return self.b2(self.b1(x, te), te)


class UNet(nn.Module):
    """Flow matching velocity field UNet (supports MNIST 1x28x28 and CIFAR-10 3x32x32)."""
    def __init__(self, in_channels: int = 1, out_channels: int = 1, width: int = 64, tdim: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        c = width
        self.register_buffer("freq", torch.exp(torch.linspace(0, math.log(1000), tdim // 2)))
        self.temb = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))

        self.down1 = Level(in_channels, c, tdim)
        self.down2 = Level(c, 2 * c, tdim)
        self.mid = Level(2 * c, 4 * c, tdim)
        self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2)
        self.dec2 = Level(4 * c, 2 * c, tdim)
        self.up1 = nn.ConvTranspose2d(2 * c, c, 2, stride=2)
        self.dec1 = Level(2 * c, c, tdim)
        self.out = nn.Conv2d(c, out_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a = t * self.freq
        te = self.temb(torch.cat([a.sin(), a.cos()], 1))
        s1 = self.down1(x, te)
        s2 = self.down2(F.max_pool2d(s1, 2), te)
        m = self.mid(F.max_pool2d(s2, 2), te)
        h = self.dec2(torch.cat([self.up2(m), s2], 1), te)
        h = self.dec1(torch.cat([self.up1(h), s1], 1), te)
        return self.out(h)


# ============================================================================
# 2. Global State & Model Cache (Load Once)
# ============================================================================

_SCRIPT_DIR = Path(__file__).resolve().parent

# Checkpoint candidates for MNIST
_MNIST_CANDIDATES = [
    _SCRIPT_DIR / "trial20-checkpoint.pt",
    _SCRIPT_DIR / "Report" / "Results" / "trial20-checkpoint.pt",
    _SCRIPT_DIR.parent / "Report" / "Results" / "trial20-checkpoint.pt",
    Path("trial20-checkpoint.pt"),
    Path("Report/Results/trial20-checkpoint.pt"),
]
DEFAULT_CKPT = str(next((p for p in _MNIST_CANDIDATES if p.exists()), _MNIST_CANDIDATES[0]))

# Checkpoint candidates for CIFAR-10
_CIFAR_CANDIDATES = [
    _SCRIPT_DIR / "cifar10-checkpoint.pt",
    _SCRIPT_DIR / "Report" / "Results" / "cifar10-checkpoint.pt",
    _SCRIPT_DIR.parent / "Report" / "Results" / "cifar10-checkpoint.pt",
    Path("cifar10-checkpoint.pt"),
    Path("Report/Results/cifar10-checkpoint.pt"),
]
DEFAULT_CIFAR_CKPT = str(next((p for p in _CIFAR_CANDIDATES if p.exists()), _CIFAR_CANDIDATES[0]))

_CLF_CANDIDATES = [
    _SCRIPT_DIR / "mnist_classifier.pt",
    _SCRIPT_DIR.parent / "mnist_classifier.pt",
    Path("mnist_classifier.pt"),
]
CLASSIFIER_PATH = str(next((p for p in _CLF_CANDIDATES if p.exists()), _CLF_CANDIDATES[0]))

_MODEL_CACHE: Dict[str, nn.Module] = {}
_MODEL_METADATA: Dict[str, str] = {}
_CLASSIFIER: Optional[torch.jit.ScriptModule] = None


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_classifier() -> Optional[torch.jit.ScriptModule]:
    global _CLASSIFIER
    if _CLASSIFIER is None and os.path.exists(CLASSIFIER_PATH):
        try:
            _CLASSIFIER = torch.jit.load(CLASSIFIER_PATH, map_location="cpu").eval()
        except Exception as e:
            print(f"[Warning] Failed to load classifier: {e}")
    return _CLASSIFIER


def load_model(ckpt_path: str = DEFAULT_CKPT) -> Tuple[UNet, str]:
    """Load model once and cache in memory. Automatically detects MNIST or CIFAR-10 channels."""
    norm_path = os.path.abspath(ckpt_path)
    dev = get_device()

    if norm_path in _MODEL_CACHE:
        model = _MODEL_CACHE[norm_path]
        first_param = next(model.parameters(), None)
        if first_param is not None and first_param.device.type != dev:
            model.to(dev)
        return model, _MODEL_METADATA.get(norm_path, "")

    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Checkpoint not found at: {norm_path}")

    print(f"[Info] Loading flow matching checkpoint from {norm_path} on {dev}...")
    ckpt = torch.load(norm_path, map_location=dev)

    if isinstance(ckpt, dict) and "net" in ckpt:
        state_dict = ckpt["net"]
        epoch = ckpt.get("epoch", "N/A")
        losses = ckpt.get("losses", [])
        last_loss = f"{np.mean(losses[-100:]):.4f}" if losses else "N/A"
    elif isinstance(ckpt, dict):
        state_dict = ckpt
        epoch, last_loss = "N/A", "N/A"
    else:
        raise ValueError("Unsupported checkpoint format.")

    # Auto-detect channels (MNIST = 1, CIFAR-10 = 3)
    in_channels = state_dict["down1.b1.c1.weight"].shape[1]
    out_channels = state_dict["out.weight"].shape[0]

    model = UNet(in_channels=in_channels, out_channels=out_channels, width=64, tdim=32).to(dev)
    model.load_state_dict(state_dict)
    model.eval()

    ds_label = "CIFAR-10 (3x32x32 RGB)" if in_channels == 3 else "MNIST (1x28x28 Grayscale)"
    meta = f"{ds_label} loaded: epoch {epoch}, loss {last_loss}, weights: {sum(p.numel() for p in model.parameters()):,} on {dev.upper()}"

    _MODEL_CACHE[norm_path] = model
    _MODEL_METADATA[norm_path] = meta
    print(f"[Success] {meta}")
    return model, meta


# Pre-load default checkpoints at startup
try:
    if os.path.exists(DEFAULT_CKPT):
        load_model(DEFAULT_CKPT)
    if os.path.exists(DEFAULT_CIFAR_CKPT):
        load_model(DEFAULT_CIFAR_CKPT)
    get_classifier()
except Exception as _e:
    print(f"[Init notice] Preload notice: {_e}")


# ============================================================================
# 3. Flow Matching Sampling Engine
# ============================================================================

@torch.no_grad()
def generate_samples(
    model: nn.Module,
    seed: int,
    step_count: int,
    solver: str = "Euler",
    noise_scale: float = 1.0,
    num_samples: int = 16,
    record_trajectory: bool = False,
) -> Tuple[torch.Tensor, Optional[List[Tuple[float, torch.Tensor]]]]:

    dev = next(model.parameters()).device
    g = torch.Generator(device=dev).manual_seed(int(seed))

    in_channels = getattr(model, "in_channels", 1)
    img_size = 32 if in_channels == 3 else 28

    # Initial condition at t=0: Gaussian noise x0 ~ N(0, sigma^2 * I)
    x = torch.randn(num_samples, in_channels, img_size, img_size, generator=g, device=dev) * float(noise_scale)

    N = max(1, int(step_count))
    time_steps = [i / N for i in range(N + 1)]

    trajectory: List[Tuple[float, torch.Tensor]] = []
    if record_trajectory:
        trajectory.append((0.0, x.clone()))

    for i in range(N):
        t_curr = time_steps[i]
        t_next = time_steps[i + 1]
        h = t_next - t_curr

        t_tensor = torch.full((num_samples, 1), t_curr, device=dev, dtype=x.dtype)
        solver_lower = solver.lower()

        if "heun" in solver_lower:
            # 2nd-order Predictor-Corrector Heun method
            v1 = model(x, t_tensor)
            if i < N - 1:
                t_tensor_next = torch.full((num_samples, 1), t_next, device=dev, dtype=x.dtype)
                x_pred = x + v1 * h
                v2 = model(x_pred, t_tensor_next)
                x = x + 0.5 * (v1 + v2) * h
            else:
                x = x + v1 * h
        else:
            # Standard 1st-order Euler method
            v = model(x, t_tensor)
            x = x + v * h

        if record_trajectory and (i == N - 1 or (i + 1) % max(1, N // 7) == 0):
            trajectory.append((t_next, x.clone()))

    return x, trajectory if record_trajectory else None


# ============================================================================
# 4. Rendering & Classifier Evaluation
# ============================================================================

def evaluate_samples_with_classifier(
    samples_net: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, float, List[int]]:
    """Evaluates generated MNIST samples with the pretrained classifier."""
    clf = get_classifier()
    u8 = ((samples_net[:, 0] + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu()

    if clf is None:
        n = len(samples_net)
        return np.ones(n), np.zeros(n, dtype=int), 1.0, [0] * 10

    with torch.no_grad():
        logits = clf(u8)
        probs = F.softmax(logits, dim=-1) if logits.max() > 1.0 or logits.min() < 0.0 else logits

    confs_t, preds_t = probs.max(dim=1)
    confs = confs_t.cpu().numpy()
    preds = preds_t.cpu().numpy()

    rec_rate = float((confs > 0.99).mean())
    counts = torch.bincount(preds_t.cpu(), minlength=10).tolist()
    return confs, preds, rec_rate, counts


def render_grid_image(
    samples_net: torch.Tensor,
    confs: Optional[np.ndarray] = None,
    preds: Optional[np.ndarray] = None,
    highlight_low_conf: bool = True,
    show_labels: bool = True,
    threshold: float = 0.99,
    scale_factor: int = 4,
) -> Image.Image:
    """Builds a high-resolution grid of images (MNIST grayscale or CIFAR-10 RGB)."""
    n = len(samples_net)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    is_cifar = (samples_net.shape[1] == 3)
    base_dim = 32 if is_cifar else 28
    cell_size = base_dim * scale_factor
    border_px = max(2, scale_factor // 2) if (highlight_low_conf and not is_cifar) else 0
    margin = 4 * scale_factor

    grid_w = cols * cell_size + (cols + 1) * margin
    grid_h = rows * cell_size + (rows + 1) * margin

    canvas = Image.new("RGB", (grid_w, grid_h), color=(18, 22, 34))
    draw = ImageDraw.Draw(canvas)

    for idx in range(n):
        r = idx // cols
        c = idx % cols
        x_pos = margin + c * (cell_size + margin)
        y_pos = margin + r * (cell_size + margin)

        if is_cifar:
            # 3-channel CIFAR-10 image [-1, 1] -> [0, 255]
            img_tensor = samples_net[idx].detach().cpu().clamp(-1.0, 1.0)
            img_u8 = ((img_tensor + 1.0) * 127.5).permute(1, 2, 0).numpy().astype(np.uint8)
            cell_img = Image.fromarray(img_u8, mode="RGB").resize(
                (cell_size, cell_size), resample=Image.Resampling.NEAREST
            )
        else:
            # 1-channel MNIST digit [-1, 1] -> [0, 255]
            digit_arr = samples_net[idx, 0].detach().cpu().clamp(-1.0, 1.0).numpy()
            digit_u8 = ((digit_arr + 1.0) * 127.5).astype(np.uint8)
            cell_img = Image.fromarray(digit_u8, mode="L").resize(
                (cell_size, cell_size), resample=Image.Resampling.NEAREST
            ).convert("RGB")

        canvas.paste(cell_img, (x_pos, y_pos))

        if not is_cifar:
            is_low_conf = False
            if confs is not None and idx < len(confs):
                if confs[idx] <= threshold:
                    is_low_conf = True

            if highlight_low_conf:
                border_color = (239, 68, 68) if is_low_conf else (34, 197, 94)
                for b in range(border_px):
                    draw.rectangle(
                        [x_pos - b, y_pos - b, x_pos + cell_size + b - 1, y_pos + cell_size + b - 1],
                        outline=border_color,
                    )

            if show_labels and preds is not None and confs is not None and idx < len(preds):
                p = preds[idx]
                conf = confs[idx]
                badge_text = f"{p} ({conf:.2f})"
                badge_bg = (239, 68, 68) if is_low_conf else (34, 197, 94)
                badge_pad = 2 * scale_factor
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None
                bbox = draw.textbbox((0, 0), badge_text, font=font) if font else (0, 0, 40, 10)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                bx1, by1 = x_pos + 2, y_pos + 2
                bx2, by2 = bx1 + tw + 2 * badge_pad, by1 + th + 2 * badge_pad
                draw.rectangle([bx1, by1, bx2, by2], fill=badge_bg)
                draw.text((bx1 + badge_pad, by1 + badge_pad), badge_text, fill=(255, 255, 255), font=font)

    return canvas


def render_trajectory_filmstrip(
    trajectory: List[Tuple[float, torch.Tensor]],
    sample_index: int = 0,
    scale_factor: int = 4,
) -> Image.Image:
    """Builds a horizontal filmstrip showing the sample evolving across time."""
    if not trajectory:
        return Image.new("RGB", (100, 100), (20, 20, 30))

    first_tensor = trajectory[0][1]
    is_cifar = (first_tensor.shape[1] == 3)
    base_dim = 32 if is_cifar else 28
    cell_size = base_dim * scale_factor
    n_frames = len(trajectory)
    margin = 4 * scale_factor
    label_h = 24

    strip_w = n_frames * cell_size + (n_frames + 1) * margin
    strip_h = cell_size + 2 * margin + label_h

    strip = Image.new("RGB", (strip_w, strip_h), color=(18, 22, 34))
    draw = ImageDraw.Draw(strip)

    for i, (t_val, tensor) in enumerate(trajectory):
        x_pos = margin + i * (cell_size + margin)
        y_pos = margin

        if is_cifar:
            arr = tensor[sample_index].detach().cpu().clamp(-1.0, 1.0)
            u8 = ((arr + 1.0) * 127.5).permute(1, 2, 0).numpy().astype(np.uint8)
            img = Image.fromarray(u8, mode="RGB").resize(
                (cell_size, cell_size), resample=Image.Resampling.NEAREST
            )
        else:
            arr = tensor[sample_index, 0].detach().cpu().clamp(-1.0, 1.0).numpy()
            u8 = ((arr + 1.0) * 127.5).astype(np.uint8)
            img = Image.fromarray(u8, mode="L").resize(
                (cell_size, cell_size), resample=Image.Resampling.NEAREST
            ).convert("RGB")

        strip.paste(img, (x_pos, y_pos))

        t_text = f"t = {t_val:.2f}"
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.text((x_pos + 4, y_pos + cell_size + 4), t_text, fill=(203, 213, 225), font=font)
        draw.rectangle([x_pos, y_pos, x_pos + cell_size - 1, y_pos + cell_size - 1], outline=(99, 102, 241), width=2)

    return strip


# ============================================================================
# 5. Core Interactive Handler
# ============================================================================

@spaces.GPU
def run_sampler(
    dataset_choice: str,
    seed: int,
    step_count: int,
    solver: str,
    noise_scale: float,
    num_samples_choice: int,
    highlight_low_conf: bool,
    show_labels: bool,
    show_trajectory: bool,
    ckpt_path: str,
    uploaded_ckpt: Optional[Any] = None,
) -> Tuple[Image.Image, Optional[Image.Image], str, str]:
    """Main callback invoked by Gradio."""
    start_time = time.time()

    is_cifar = "cifar" in dataset_choice.lower()

    # Determine which checkpoint to use
    if uploaded_ckpt is not None:
        target_path = uploaded_ckpt.name if hasattr(uploaded_ckpt, "name") else str(uploaded_ckpt)
    elif is_cifar:
        target_path = DEFAULT_CIFAR_CKPT
    else:
        target_path = ckpt_path if ckpt_path else DEFAULT_CKPT

    # 1. Fetch model from cache (load once)
    try:
        model, meta = load_model(target_path)
    except Exception as e:
        err_msg = f"Error loading checkpoint: {str(e)}"
        empty_img = Image.new("RGB", (200, 200), color=(40, 20, 20))
        return empty_img, None, err_msg, ""

    # Ensure model is on active device
    dev = get_device()
    first_p = next(model.parameters(), None)
    if first_p is not None and first_p.device.type != dev:
        model.to(dev)

    # 2. Generate samples via ODE integration
    try:
        samples_net, trajectory = generate_samples(
            model=model,
            seed=int(seed),
            step_count=int(step_count),
            solver=solver,
            noise_scale=float(noise_scale),
            num_samples=int(num_samples_choice),
            record_trajectory=bool(show_trajectory),
        )
    except Exception as e:
        err_msg = f"Error during sampling: {str(e)}"
        empty_img = Image.new("RGB", (200, 200), color=(40, 20, 20))
        return empty_img, None, err_msg, ""

    latency_ms = (time.time() - start_time) * 1000.0

    # 3. Diagnostics
    confs, preds = None, None
    if not is_cifar and samples_net.shape[1] == 1:
        confs, preds, rec_rate, counts = evaluate_samples_with_classifier(samples_net)
        low_conf_count = int(np.sum(confs <= 0.99))
        metrics_markdown = f"""
### Sampling Diagnostics (MNIST)
- **Model**: `MNIST Flow Matching (Trial 20)`
- **Recognition Score (Conf > 0.99)**: `{rec_rate * 100:.1f}%`
- **Low-Confidence Samples (<= 0.99)**: `{low_conf_count} / {len(samples_net)}` *(red borders in grid)*
- **Integration Latency**: `{latency_ms:.1f} ms` on `{get_device().upper()}`
- **Sampler**: `{solver}` with `{step_count}` steps | Noise σ = `{noise_scale}`
- **Digit Counts [0-9]**: `{counts}`
"""
    else:
        grid_dim = int(math.isqrt(len(samples_net)))
        metrics_markdown = f"""
### Sampling Diagnostics (CIFAR-10)
- **Model**: `CIFAR-10 Flow Matching U-Net`
- **Channels & Resolution**: `3 channels (RGB) · 32 x 32`
- **Integration Latency**: `{latency_ms:.1f} ms` on `{get_device().upper()}`
- **Sampler**: `{solver}` with `{step_count}` steps | Noise σ = `{noise_scale}`
- **Samples Grid**: `{len(samples_net)}` images ({grid_dim}x{grid_dim})
"""

    # 4. Render main grid
    grid_img = render_grid_image(
        samples_net=samples_net,
        confs=confs,
        preds=preds,
        highlight_low_conf=highlight_low_conf,
        show_labels=show_labels,
        threshold=0.99,
        scale_factor=4,
    )

    # 5. Render trajectory strip if requested
    traj_img = None
    if show_trajectory and trajectory:
        traj_img = render_trajectory_filmstrip(trajectory, sample_index=0, scale_factor=4)

    status_badge = f"Generated {len(samples_net)} samples in {latency_ms:.1f}ms ({solver}, N={step_count}, Seed={seed}) on {dataset_choice.split()[0]}"
    return grid_img, traj_img, metrics_markdown, status_badge


def reload_checkpoint_handler(new_path: str) -> str:
    """Manually load/reload another checkpoint into cache."""
    try:
        _, meta = load_model(new_path)
        return f"{meta}"
    except Exception as e:
        return f"Failed to load checkpoint: {str(e)}"


# ============================================================================
# 6. Gradio User Interface
# ============================================================================

CUSTOM_CSS = """
/* Modern Dark Glassmorphism Theme */
body, .gradio-container {
    background-color: #0b0f19 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}
.header-box {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
    padding: 24px;
    border-radius: 14px;
    color: white;
    margin-bottom: 20px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.1);
}
.control-card {
    background: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
}
.output-card {
    background: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
"""

def create_ui() -> gr.Blocks:
    with gr.Blocks(title="Flow Matching Interactive Sampler") as demo:
        # Header
        gr.HTML("""
        <div class="header-box">
            <h1 style="margin: 0; font-size: 28px; font-weight: 700; display: flex; align-items: center; gap: 10px;">
                🌊 Flow Matching Interactive Sampler
            </h1>
            <p style="margin: 6px 0 0 0; opacity: 0.9; font-size: 14px;">
                Simulate continuous velocity fields using Euler and Heun ODE solvers across MNIST Digits and CIFAR-10 Color Images.
            </p>
        </div>
        """)

        with gr.Row():
            # ----------------- Left: Control Panel -----------------
            with gr.Column(scale=4, elem_classes=["control-card"]):
                gr.Markdown("### Sampler Controls")

                # Dataset Selector
                dataset_radio = gr.Radio(
                    choices=["MNIST (28x28 Grayscale)", "CIFAR-10 (32x32 RGB)"],
                    value="MNIST (28x28 Grayscale)",
                    label="Select Dataset / Model",
                    info="Switch between MNIST Handwritten Digits and CIFAR-10 Natural Color Images",
                )

                # Checkpoint Accordion
                with gr.Accordion("📁 Model Checkpoint Settings", open=False):
                    ckpt_upload = gr.File(
                        file_count="single",
                        file_types=[".pt"],
                        label="Upload Custom Checkpoint (.pt)",
                    )
                    ckpt_input = gr.Textbox(
                        value=DEFAULT_CKPT,
                        label="Default Checkpoint Path (.pt)",
                        info="Points to the trained flow matching weights",
                    )
                    with gr.Row():
                        reload_btn = gr.Button("Reload Checkpoint", variant="secondary", size="sm")
                    ckpt_status = gr.Markdown(
                        value=f"Device: `{get_device().upper()}` | Initial checkpoint: `{Path(DEFAULT_CKPT).name}`"
                    )

                # Core Controls
                with gr.Group():
                    gr.Markdown("#### Core Parameters")
                    with gr.Row():
                        seed_slider = gr.Slider(
                            minimum=0,
                            maximum=99999,
                            value=42,
                            step=1,
                            label="Random Seed",
                            info="Controls initial noise x_0",
                        )
                        randomize_btn = gr.Button("🎲 Random", size="sm", elem_id="rand_btn")

                    step_slider = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=25,
                        step=1,
                        label="Step Count (N)",
                        info="Number of integration steps from t=0 to t=1",
                    )

                # Additional Model Properties
                with gr.Group():
                    gr.Markdown("#### Additional Model Properties")
                    solver_dropdown = gr.Dropdown(
                        choices=["Euler", "Heun"],
                        value="Euler",
                        label="Sampler",
                    )

                    noise_slider = gr.Slider(
                        minimum=0.1,
                        maximum=2.5,
                        value=1.0,
                        step=0.05,
                        label="Noise Temperature (σ)",
                        info="Scale factor for initial Gaussian noise x_0 ~ N(0, σ²I)",
                    )

                    samples_dropdown = gr.Dropdown(
                        choices=[1, 4, 9, 16, 25, 36, 64],
                        value=16,
                        label="Number of Samples (Grid Size)",
                        info="Generates 1x1, 2x2, 3x3, 4x4, 5x5, 6x6, or 8x8 grid",
                    )

                # Diagnostic Options
                with gr.Group():
                    gr.Markdown("#### Diagnostics & Visuals")
                    with gr.Row():
                        highlight_box = gr.Checkbox(
                            value=True,
                            label="Highlight Low Confidence (MNIST)",
                            info="Red border if classifier conf ≤ 0.99",
                        )
                        labels_box = gr.Checkbox(
                            value=True,
                            label="Show Predicted Digit Badges (MNIST)",
                            info="Overlays digit & confidence",
                        )
                    trajectory_box = gr.Checkbox(
                        value=True,
                        label="Record Reverse Flow Trajectory",
                        info="Generates evolution filmstrip from t=0.0 to t=1.0",
                    )

                # Submit Button
                generate_btn = gr.Button("Generate Samples", variant="primary", size="lg")

            # ----------------- Right: Results & Visualizations -----------------
            with gr.Column(scale=6, elem_classes=["output-card"]):
                status_box = gr.Markdown("Ready to sample. Press **Generate Samples**.")

                with gr.Tabs():
                    with gr.TabItem("Generated Samples Grid"):
                        gr.Markdown(
                            "<p style='margin: 4px 0 6px 0; font-size: 13px; color: #94a3b8;'>"
                            "Generated Samples &bull; MNIST Digits or CIFAR-10 RGB"
                            "</p>"
                        )
                        grid_output = gr.Image(
                            type="pil",
                            interactive=False,
                            show_label=False,
                        )

                    with gr.TabItem("Flow Trajectory (t: 0 → 1)"):
                        gr.Markdown(
                            "<p style='margin: 4px 0 6px 0; font-size: 13px; color: #94a3b8;'>"
                            "Reverse Vector Field Trajectory Evolution (t = 0 Noise &rarr; t = 1 Sample)"
                            "</p>"
                        )
                        traj_output = gr.Image(
                            type="pil",
                            interactive=False,
                            show_label=False,
                        )

                metrics_output = gr.Markdown(
                    value="""
### Sampling Diagnostics
- Click **Generate Samples** to evaluate.
                    """
                )

        # Event Handlers
        def randomize_seed():
            import random
            return random.randint(0, 99999)

        randomize_btn.click(fn=randomize_seed, outputs=seed_slider)

        def on_dataset_change(choice):
            if "cifar" in choice.lower():
                return DEFAULT_CIFAR_CKPT, f"Device: `{get_device().upper()}` | Selected: `CIFAR-10`"
            else:
                return DEFAULT_CKPT, f"Device: `{get_device().upper()}` | Selected: `MNIST`"

        dataset_radio.change(
            fn=on_dataset_change,
            inputs=[dataset_radio],
            outputs=[ckpt_input, ckpt_status],
        )

        def handle_ckpt_upload(uploaded_file):
            if not uploaded_file:
                return f"Device: `{get_device().upper()}` | Using default: `{Path(DEFAULT_CKPT).name}`"
            path = uploaded_file.name if hasattr(uploaded_file, "name") else str(uploaded_file)
            try:
                _, meta = load_model(path)
                return f"✅ Loaded custom checkpoint: `{Path(path).name}` | {meta}"
            except Exception as e:
                return f"❌ Failed to load custom checkpoint: {e}"

        ckpt_upload.change(
            fn=handle_ckpt_upload,
            inputs=[ckpt_upload],
            outputs=[ckpt_status],
        )

        reload_btn.click(
            fn=reload_checkpoint_handler,
            inputs=[ckpt_input],
            outputs=[ckpt_status],
        )

        sampler_inputs = [
            dataset_radio,
            seed_slider,
            step_slider,
            solver_dropdown,
            noise_slider,
            samples_dropdown,
            highlight_box,
            labels_box,
            trajectory_box,
            ckpt_input,
            ckpt_upload,
        ]

        generate_btn.click(
            fn=run_sampler,
            inputs=sampler_inputs,
            outputs=[
                grid_output,
                traj_output,
                metrics_output,
                status_box,
            ],
        )

        # Trigger generation on load
        demo.load(
            fn=run_sampler,
            inputs=sampler_inputs,
            outputs=[
                grid_output,
                traj_output,
                metrics_output,
                status_box,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    theme = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")
    port = int(os.environ.get("PORT", 7860))
    share = os.environ.get("GRADIO_SHARE", "False").lower() in ("true", "1")
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=theme,
        css=CUSTOM_CSS,
        share=share,
    )
