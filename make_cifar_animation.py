import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {dev}")

# 1. Define CIFAR-10 UNet Architecture matching training
class Block(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.norm = nn.GroupNorm(4, cout)
        self.t = nn.Linear(tdim, cout)
        self.proj = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x, te):
        h = self.norm(self.c1(x)) + self.t(te)[:, :, None, None]
        change = self.c2(F.silu(h))
        return self.proj(x) + change

class Level(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.b1, self.b2 = Block(cin, cout, tdim), Block(cout, cout, tdim)

    def forward(self, x, te):
        return self.b2(self.b1(x, te), te)

class UNet(nn.Module):
    def __init__(self, width=64, tdim=32):
        super().__init__()
        c = width
        self.register_buffer("freq", torch.exp(torch.linspace(0, math.log(1000), tdim // 2)))
        self.temb = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))

        self.down1 = Level(3, c, tdim)             # 32 x 32 (3 RGB channels)
        self.down2 = Level(c, 2 * c, tdim)         # 16 x 16
        self.mid = Level(2 * c, 4 * c, tdim)       # 8 x 8
        self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2)
        self.dec2 = Level(4 * c, 2 * c, tdim)
        self.up1 = nn.ConvTranspose2d(2 * c, c, 2, stride=2)
        self.dec1 = Level(2 * c, c, tdim)
        self.out = nn.Conv2d(c, 3, 1)              # 3 RGB channels

    def forward(self, x, t):
        a = t * self.freq
        te = self.temb(torch.cat([a.sin(), a.cos()], 1))
        s1 = self.down1(x, te)
        s2 = self.down2(F.max_pool2d(s1, 2), te)
        m = self.mid(F.max_pool2d(s2, 2), te)
        h = self.dec2(torch.cat([self.up2(m), s2], 1), te)
        h = self.dec1(torch.cat([self.up1(h), s1], 1), te)
        return self.out(h)

# 2. Load Checkpoint
CKPT = "Report/Results/cifar10-checkpoint.pt"
net = UNet().to(dev)
ck = torch.load(CKPT, map_location=dev)
net.load_state_dict(ck["net"])
net.eval()
print(f"Loaded CIFAR-10 checkpoint from {CKPT} (Epoch {ck.get('epoch', 'N/A')})")

# 3. Simulate Flow Trajectory with Heun Solver
N_STEPS = 50
NUM_SAMPLES = 16  # 4x4 grid
SEED = 42

torch.manual_seed(SEED)
x = torch.randn(NUM_SAMPLES, 3, 32, 32, device=dev)

trajectory = [x.clone()]  # t=0

with torch.no_grad():
    for i in range(N_STEPS):
        t = torch.full((len(x), 1), i / N_STEPS, device=dev)
        v = net(x, t)
        if i < N_STEPS - 1:
            v2 = net(x + v / N_STEPS, t + 1 / N_STEPS)
            v = (v + v2) / 2
        x = x + v / N_STEPS
        trajectory.append(x.clone())

print(f"Computed trajectory: {len(trajectory)} timesteps from t=0.0 to t=1.0")

# 4. Generate Animated GIF
gif_frames = []
for idx, x_t in enumerate(trajectory):
    t_val = idx / N_STEPS
    # Normalize to [0, 1]
    rgb = ((x_t + 1.0) * 0.5).clamp(0, 1)
    # Make a 4x4 grid
    grid = torchvision.utils.make_grid(rgb, nrow=4, padding=2, pad_value=0.15)
    # Convert to PIL Image
    grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(grid_np)
    # Resize for higher display resolution (e.g. 512x512)
    img_large = img.resize((512, 512), Image.Resampling.NEAREST)
    
    # Overlay timestep label banner
    banner = Image.new("RGB", (512, 36), (25, 25, 30))
    draw = ImageDraw.Draw(banner)
    text = f"Integration Step: {idx}/{N_STEPS}   |   t = {t_val:.2f}"
    draw.text((12, 10), text, fill=(240, 240, 250))
    
    combined = Image.new("RGB", (512, 548), (25, 25, 30))
    combined.paste(banner, (0, 0))
    combined.paste(img_large, (0, 36))
    
    gif_frames.append(combined)

# Add a pause at the end (repeat final frame 20 times ~ 1 second)
final_frame = gif_frames[-1]
for _ in range(20):
    gif_frames.append(final_frame)

out_gif = "Report/Results/cifar10_generation_animation.gif"
gif_frames[0].save(
    out_gif,
    save_all=True,
    append_images=gif_frames[1:],
    duration=60,  # ~16 fps
    loop=0
)
print(f"Saved animated GIF to {out_gif} ({os.path.getsize(out_gif) / 1024:.1f} KB)")

# 5. Generate Publication-Grade Representative Still (Filmstrip) for Report
# Show 4 selected diverse samples across 6 timesteps: t = 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
milestone_indices = [0, 10, 20, 30, 40, 50]
t_labels = ["t = 0.0\n(Noise)", "t = 0.2\n(Color/Hues)", "t = 0.4\n(Coarse Mass)", "t = 0.6\n(Silhouette)", "t = 0.8\n(Features)", "t = 1.0\n(Final Image)"]

selected_samples = [0, 2, 5, 8]  # Pick 4 diverse samples
fig, axes = plt.subplots(len(selected_samples), len(milestone_indices), figsize=(12, 8.5), dpi=300)

for r, s_idx in enumerate(selected_samples):
    for c, m_idx in enumerate(milestone_indices):
        ax = axes[r, c]
        img_tensor = trajectory[m_idx][s_idx]
        img_rgb = ((img_tensor + 1.0) * 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        ax.imshow(img_rgb, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Style borders
        for spine in ax.spines.values():
            spine.set_color("#4A5568")
            spine.set_linewidth(1.2)
            
        if r == 0:
            ax.set_title(t_labels[c], fontsize=11, fontweight="bold", pad=8, color="#1A202C")
        if c == 0:
            ax.set_ylabel(f"Sample {r + 1}", fontsize=11, fontweight="bold", color="#1A202C", labelpad=8)

plt.subplots_adjust(wspace=0.08, hspace=0.08, top=0.90, bottom=0.05, left=0.08, right=0.96)
fig.suptitle("CIFAR-10 Sample Evolution Across Integration Time (Heun Solver, $N=50$)", fontsize=13, fontweight="bold", y=0.97)

out_still = "Report/Results/cifar10_animation_still.png"
plt.savefig(out_still, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved representative still filmstrip to {out_still} ({os.path.getsize(out_still) / 1024:.1f} KB)")
print("Done!")
