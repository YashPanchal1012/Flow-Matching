import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    """Two 3x3 convolutions with time offset added after the first."""
    def __init__(self, cin: int, cout: int, tdim: int):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)

    def forward(self, x: torch.Tensor, te: torch.Tensor) -> torch.Tensor:
        h = self.c1(x) + self.t(te)[:, :, None, None]
        return F.relu(self.c2(F.relu(h)))


class UNet(nn.Module):
    """Flow matching velocity field UNet for 28x28 grayscale images."""
    def __init__(self, width: int = 16, tdim: int = 32):
        super().__init__()
        c = width
        self.register_buffer("freq", torch.exp(torch.linspace(0, math.log(1000), tdim // 2)))
        self.temb = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.down1 = Block(1, c, tdim)                    # 28 x 28
        self.down2 = Block(c, 2 * c, tdim)                # 14 x 14
        self.mid = Block(2 * c, 4 * c, tdim)              # 7 x 7
        self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2)
        self.dec2 = Block(4 * c, 2 * c, tdim)
        self.up1 = nn.ConvTranspose2d(2 * c, c, 2, stride=2)
        self.dec1 = Block(2 * c, c, tdim)
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: B x 1 x 28 x 28, t: B x 1
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

DEFAULT_CKPT = r"C:\MyFiles\UMN Notes\CSCI 5561\Flow-Matching\flow_matching_examples\checkpoints\checkpoint.pt"
CLASSIFIER_PATH = r"mnist_classifier.pt"

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL_CACHE: Dict[str, nn.Module] = {}
_MODEL_METADATA: Dict[str, str] = {}
_CLASSIFIER: Optional[torch.jit.ScriptModule] = None


def get_device() -> str:
    return _DEVICE


def get_classifier() -> Optional[torch.jit.ScriptModule]:
    global _CLASSIFIER
    if _CLASSIFIER is None and os.path.exists(CLASSIFIER_PATH):
        try:
            _CLASSIFIER = torch.jit.load(CLASSIFIER_PATH, map_location="cpu").eval()
        except Exception as e:
            print(f"[Warning] Failed to load classifier: {e}")
    return _CLASSIFIER


def load_model(ckpt_path: str = DEFAULT_CKPT) -> Tuple[UNet, str]:
    """Load model once and cache in memory. Respond to queries without retraining."""
    norm_path = os.path.abspath(ckpt_path)
    dev = get_device()

    if norm_path in _MODEL_CACHE:
        return _MODEL_CACHE[norm_path], _MODEL_METADATA.get(norm_path, "")

    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Checkpoint not found at: {norm_path}")

    print(f"[Info] Loading flow matching checkpoint from {norm_path} on {dev} (one-time load)...")
    ckpt = torch.load(norm_path, map_location=dev)
    model = UNet(width=16, tdim=32).to(dev)

    if isinstance(ckpt, dict) and "net" in ckpt:
        model.load_state_dict(ckpt["net"])
        epoch = ckpt.get("epoch", "N/A")
        losses = ckpt.get("losses", [])
        last_loss = f"{np.mean(losses[-100:]):.4f}" if losses else "N/A"
        meta = f"Checkpoint loaded: epoch {epoch}, final loss {last_loss}, weights: {sum(p.numel() for p in model.parameters()):,} on {dev.upper()}"
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
        meta = f"State dict loaded: {sum(p.numel() for p in model.parameters()):,} weights on {dev.upper()}"
    else:
        raise ValueError("Unsupported checkpoint format.")

    model.eval()
    _MODEL_CACHE[norm_path] = model
    _MODEL_METADATA[norm_path] = meta
    print(f"[Success] {meta}")
    return model, meta


# Pre-load default checkpoint at module import time
try:
    load_model(DEFAULT_CKPT)
    get_classifier()
except Exception as _e:
    print(f"[Init notice] Default checkpoint could not be auto-loaded: {_e}")


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
    time_schedule_gamma: float = 1.0,
    num_samples: int = 16,
    record_trajectory: bool = False,
) -> Tuple[torch.Tensor, Optional[List[Tuple[float, torch.Tensor]]]]:

    dev = next(model.parameters()).device
    g = torch.Generator(device=dev).manual_seed(int(seed))

    # Initial condition at t=0: Gaussian noise x0 ~ N(0, sigma^2 * I)
    x = torch.randn(num_samples, 1, 28, 28, generator=g, device=dev) * float(noise_scale)

    N = max(1, int(step_count))
    gamma = float(time_schedule_gamma)

    # Discretization grid in [0, 1]
    time_steps = [(i / N) ** gamma for i in range(N + 1)]

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
    """
    Evaluates generated samples with the course's pretrained classifier.
    Returns:
        confs: Confidence values for top predicted digit
        preds: Predicted digit classes (0-9)
        recognition_rate: Percentage of samples with confidence > 0.99
        bincount: Count of each digit 0-9
    """
    clf = get_classifier()
    # Convert from [-1, 1] to uint8 [0, 255]
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
    """
    Builds a high-resolution grid of MNIST digits.
    Low-confidence digits (conf <= threshold) are bordered in red.
    """
    n = len(samples_net)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    # Base image dimensions (28x28 + borders)
    cell_size = 28 * scale_factor
    border_px = max(2, scale_factor // 2) if highlight_low_conf else 0
    margin = 4 * scale_factor

    grid_w = cols * cell_size + (cols + 1) * margin
    grid_h = rows * cell_size + (rows + 1) * margin

    # Dark slate background for premium look
    canvas = Image.new("RGB", (grid_w, grid_h), color=(18, 22, 34))
    draw = ImageDraw.Draw(canvas)

    for idx in range(n):
        r = idx // cols
        c = idx % cols
        x_pos = margin + c * (cell_size + margin)
        y_pos = margin + r * (cell_size + margin)

        # Single digit array
        digit_arr = samples_net[idx, 0].detach().cpu().clamp(-1.0, 1.0).numpy()
        digit_u8 = ((digit_arr + 1.0) * 127.5).astype(np.uint8)

        # Upscale with nearest neighbor for sharp pixel grid
        digit_img = Image.fromarray(digit_u8, mode="L").resize(
            (cell_size, cell_size), resample=Image.Resampling.NEAREST
        ).convert("RGB")

        canvas.paste(digit_img, (x_pos, y_pos))

        # Check confidence for highlight border
        is_low_conf = False
        if confs is not None and idx < len(confs):
            if confs[idx] <= threshold:
                is_low_conf = True

        if highlight_low_conf:
            border_color = (239, 68, 68) if is_low_conf else (34, 197, 94)  # Red vs Green
            # Draw border
            for b in range(border_px):
                draw.rectangle(
                    [x_pos - b, y_pos - b, x_pos + cell_size + b - 1, y_pos + cell_size + b - 1],
                    outline=border_color,
                )

        # Draw predicted label badge
        if show_labels and preds is not None and confs is not None and idx < len(preds):
            p = preds[idx]
            cf = confs[idx]
            badge_text = f"{p} ({cf:.2f})"
            # Small background box
            draw.rectangle(
                [x_pos + 2, y_pos + cell_size - 16, x_pos + 48, y_pos + cell_size - 2],
                fill=(0, 0, 0, 180),
            )
            draw.text(
                (x_pos + 4, y_pos + cell_size - 15),
                badge_text,
                fill=(255, 255, 255),
            )

    return canvas


def render_trajectory_filmstrip(
    trajectory: List[Tuple[float, torch.Tensor]],
    sample_index: int = 0,
    scale_factor: int = 4,
) -> Image.Image:
    """
    Renders horizontal evolution of a selected sample from noise (t=0) to digit (t=1).
    """
    if not trajectory:
        return Image.new("RGB", (100, 100), color=(20, 20, 20))

    cell_size = 28 * scale_factor
    margin = 8
    header_h = 24
    num_snaps = len(trajectory)

    strip_w = num_snaps * cell_size + (num_snaps + 1) * margin
    strip_h = cell_size + margin * 2 + header_h

    strip = Image.new("RGB", (strip_w, strip_h), color=(24, 28, 42))
    draw = ImageDraw.Draw(strip)

    for i, (t_val, snapshot) in enumerate(trajectory):
        idx = min(sample_index, len(snapshot) - 1)
        x_pos = margin + i * (cell_size + margin)
        y_pos = margin + header_h

        img_arr = snapshot[idx, 0].detach().cpu().clamp(-1.0, 1.0).numpy()
        img_u8 = ((img_arr + 1.0) * 127.5).astype(np.uint8)
        img_pil = Image.fromarray(img_u8, mode="L").resize(
            (cell_size, cell_size), resample=Image.Resampling.NEAREST
        ).convert("RGB")

        strip.paste(img_pil, (x_pos, y_pos))

        # Time label
        label = f"t = {t_val:.2f}"
        draw.text((x_pos + 4, 4), label, fill=(190, 200, 220))

        # Accent border
        draw.rectangle(
            [x_pos, y_pos, x_pos + cell_size - 1, y_pos + cell_size - 1],
            outline=(79, 70, 229),
            width=2,
        )

    return strip


# ============================================================================
# 5. Core Interactive Handler
# ============================================================================

def run_sampler(
    seed: int,
    step_count: int,
    solver: str,
    noise_scale: float,
    time_schedule_gamma: float,
    num_samples_choice: int,
    highlight_low_conf: bool,
    show_labels: bool,
    show_trajectory: bool,
    ckpt_path: str,
) -> Tuple[Image.Image, Optional[Image.Image], str, str]:
    """
    Main callback invoked by Gradio. Loads checkpoint once, generates samples,
    and returns visualized grid, metrics, and optional trajectory filmstrip.
    """
    start_time = time.time()

    # 1. Fetch model from cache (load once)
    try:
        model, meta = load_model(ckpt_path)
    except Exception as e:
        err_msg = f"Error loading checkpoint: {str(e)}"
        empty_img = Image.new("RGB", (200, 200), color=(40, 20, 20))
        return empty_img, None, err_msg, ""

    # 2. Generate samples via ODE integration
    try:
        samples_net, trajectory = generate_samples(
            model=model,
            seed=int(seed),
            step_count=int(step_count),
            solver=solver,
            noise_scale=float(noise_scale),
            time_schedule_gamma=float(time_schedule_gamma),
            num_samples=int(num_samples_choice),
            record_trajectory=bool(show_trajectory),
        )
    except Exception as e:
        err_msg = f"Error during sampling: {str(e)}"
        empty_img = Image.new("RGB", (200, 200), color=(40, 20, 20))
        return empty_img, None, err_msg, ""

    latency_ms = (time.time() - start_time) * 1000.0

    # 3. Evaluate with pretrained course classifier
    confs, preds, rec_rate, counts = evaluate_samples_with_classifier(samples_net)

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

    # 6. Format metrics
    low_conf_count = int(np.sum(confs <= 0.99))
    full_cov = all(c >= 1 for c in counts) if len(samples_net) >= 10 else "N/A"

    metrics_markdown = f"""
### Sampling Diagnostics
- **Recognition Score (Conf > 0.99)**: `{rec_rate * 100:.1f}%`
- **Low-Confidence Samples (<= 0.99)**: `{low_conf_count} / {len(samples_net)}` *(red borders in grid)*
- **Integration Latency**: `{latency_ms:.1f} ms` on `{get_device().upper()}`
- **ODE Solver**: `{solver}` with `{step_count}` steps | **Noise $\sigma$**: `{noise_scale}`
- **Digit Counts [0-9]**: `{counts}`
    """

    status_badge = f"Generated {len(samples_net)} samples in {latency_ms:.1f}ms ({solver}, N={step_count}, Seed={seed})"
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
.stat-box {
    background: #1f2937;
    border-radius: 8px;
    padding: 12px;
    border-left: 4px solid #6366f1;
}
"""

def create_ui() -> gr.Blocks:
    with gr.Blocks(title="Flow Matching Interactive Sampler") as demo:
        # Header
        gr.HTML("""
        <div class="header-box">
            <h1 style="margin: 0; font-size: 28px; font-weight: 700; display: flex; align-items: center; gap: 10px;">
                Flow Matching Interactive Sampler
            </h1>
            <p style="margin: 8px 0 0 0; opacity: 0.85; font-size: 15px;">
                CSCI 5561 Generative AI for Images &bull; Interactive ODE Sampling & Diagnostic Suite
            </p>
        </div>
        """)

        with gr.Row():
            # ----------------- Left: Control Panel -----------------
            with gr.Column(scale=4, elem_classes=["control-card"]):
                gr.Markdown("### Sampler Controls")

                # Checkpoint Accordion
                with gr.Accordion("📁 Model Checkpoint Settings", open=False):
                    ckpt_input = gr.Textbox(
                        value=DEFAULT_CKPT,
                        label="Checkpoint Path (.pt)",
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

                    # schedule_slider = gr.Slider(
                    #     minimum=0.2,
                    #     maximum=3.0,
                    #     value=1.0,
                    #     step=0.1,
                    #     label="Time Warping Exponent (γ)",
                    #     info="t_i = (i / N)^γ (1.0 = uniform; >1.0 concentrates steps near clean data)",
                    # )

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
                            label="Highlight Low Confidence (≤ 0.99)",
                            info="Red border if classifier conf ≤ 0.99",
                        )
                        labels_box = gr.Checkbox(
                            value=True,
                            label="Show Predicted Digit Badges",
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
                            "Generated MNIST Digits &bull; <span style='color: #ef4444; font-weight: 600;'>Red border</span> indicates classifier confidence &le; 0.99"
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
                            "Reverse Vector Field Trajectory Evolution ($t = 0.00$ Noise &rarr; $t = 1.00$ Digit)"
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
- **Recognition Score (Conf > 0.99)**: `--%`
- **Low-Confidence Samples**: `--`
- **Integration Latency**: `-- ms`
- Click **Generate Samples** to evaluate.
                    """
                )

        # Event Handlers
        def randomize_seed():
            import random
            return random.randint(0, 99999)

        randomize_btn.click(fn=randomize_seed, outputs=seed_slider)

        reload_btn.click(
            fn=reload_checkpoint_handler,
            inputs=[ckpt_input],
            outputs=[ckpt_status],
        )

        generate_btn.click(
            fn=run_sampler,
            inputs=[
                seed_slider,
                step_slider,
                solver_dropdown,
                noise_slider,
                schedule_slider,
                samples_dropdown,
                highlight_box,
                labels_box,
                trajectory_box,
                ckpt_input,
            ],
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
            inputs=[
                seed_slider,
                step_slider,
                solver_dropdown,
                noise_slider,
                schedule_slider,
                samples_dropdown,
                highlight_box,
                labels_box,
                trajectory_box,
                ckpt_input,
            ],
            outputs=[
                grid_output,
                traj_output,
                metrics_output,
                status_box,
            ],
        )

    return demo


if __name__ == "__main__":
    # Standalone entrypoint
    demo = create_ui()
    theme = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")
    demo.launch(
        server_name="127.0.0.1",
        server_port=8000,
        theme=theme,
        css=CUSTOM_CSS,
        share=False,
    )
