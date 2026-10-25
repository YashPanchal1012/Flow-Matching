# Flow Matching

## 1. Installation

```bash
conda create -n flow_matching python=3.10 -y
conda activate flow_matching
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib pillow gradio tqdm
```

## 2. Training Models

Run the respective cells in `main.ipynb`:

* **2D Toy Model (Dog Face):**
  ```python
  flow_matching_2D()
  ```

* **Image U-Net Model (MNIST):**
  ```python
  net = UNet(width=64, tdim=32).to(dev)
  net, losses = train_until(net, last_epoch=120)
  ```

## 3. Loading a Checkpoint

```python
import torch

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = UNet(width=64, tdim=32).to(dev)

ckpt = torch.load("Report/Results/trial20-checkpoint.pt", map_location=dev)
net.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
net.eval()
```

## 4. Generating Samples

### Programmatic (Euler Sampler)

```python
@torch.no_grad()
def sample(net, n=16, steps=25):
    x = torch.randn(n, 1, 28, 28, device=dev)
    for i in range(steps):
        t = torch.full((n, 1), i / steps, device=dev)
        x = x + net(x, t) * (1.0 / steps)
    return x.clamp(-1.0, 1.0)

samples = sample(net)
```

### Interactive Web UI (Gradio)

```bash
python interactive_sampler.py
```
Open the local URL (`http://127.0.0.1:7860`) in your browser to interactively generate digits, inspect classifier scores, and visualize trajectories.
