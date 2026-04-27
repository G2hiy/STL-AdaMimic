"""训练根平移扩散模型 (创新点① root-only).

输入:
    --data_path  S1 产物 amass_g1_root.pt
                 {trajectories[N,T,3], norm_stats{mean,std}, source_ids, meta}
    --ref_path   基准 data.pt, 用于记录 ref_T / 与训练 seq_len 对齐诊断

输出:
    --ckpt_dir 下 epoch_xxxx.pt, 含 model.state_dict() + scheduler config + norm_stats + meta

设计:
    - traj_dim = 3 (仅 base_pos)
    - cond_dim = 3 (条件 Δp ∈ ℝ^3, 取自归一化空间 base_pos 末-初位移)
    - Checkpoint 额外存 norm_stats (generate 时需反归一化)

用法:
    python scripts/train_diffusion_full.py \\
        --data_path resources/dataset/diffusion_train/amass_g1_root.pt \\
        --ref_path  legged_gym/resources/dataset/g1_dof27_data/high_jump/output/data.pt \\
        --ckpt_dir  checkpoints/diffusion_root/high_jump \\
        --epochs 200 --batch_size 64 --lr 2e-4 --d_model 256 --n_layers 6
"""

import argparse
import os
import time
import torch
from torch.utils.data import DataLoader, Dataset
from diffusers import DDPMScheduler

from legged_gym.diffusion.root_mdm import RootDiffusionModel


TRAJ_DIM = 3        # base_pos


class RootTrajectoryFullDataset(Dataset):
    """3-dim 归一化 base_pos 轨迹数据集.

    __getitem__ 返回 (x[T,3], delta_p[3])
        delta_p = x[-1] - x[0]    (归一化空间 base_pos 位移)
    弱监督条件; generate 阶段条件由用户/Δp 采样器提供.
    """

    def __init__(self, trajectories: torch.Tensor, cond_jitter: float = 0.05):
        assert trajectories.dim() == 3 and trajectories.shape[-1] == TRAJ_DIM, \
            f"expected (N,T,{TRAJ_DIM}), got {tuple(trajectories.shape)}"
        self.traj = trajectories.float()
        self.cond_jitter = cond_jitter

    def __len__(self):
        return self.traj.shape[0]

    def __getitem__(self, idx):
        x = self.traj[idx]                       # (T, 3)
        delta_p = x[-1] - x[0]                   # (3,) 归一化空间
        if self.cond_jitter > 0:
            delta_p = delta_p + torch.randn_like(delta_p) * self.cond_jitter
        return x, delta_p


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--ref_path", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num_train_timesteps", type=int, default=1000)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--cond_jitter", type=float, default=0.05)
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    blob = torch.load(args.data_path)
    assert isinstance(blob, dict) and "trajectories" in blob and "norm_stats" in blob, \
        "data_path 必须是 prepare_diffusion_full_training_set.py 的产物 (含 norm_stats)"
    trajectories = blob["trajectories"]
    norm_stats = blob["norm_stats"]
    N, T, D = trajectories.shape
    assert D == TRAJ_DIM, f"traj_dim must be {TRAJ_DIM}, got {D}"
    print(f"[data] N={N}  T={T}  D={D}")

    ref = torch.load(args.ref_path)
    ref_T = ref["base_position"].shape[0]
    if T != ref_T:
        print(f"[warn] training T={T} != ref_T={ref_T}; generate 时会按 ref_T 对齐")

    dataset = RootTrajectoryFullDataset(trajectories, cond_jitter=args.cond_jitter)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True)

    model_config = dict(
        traj_dim=TRAJ_DIM,
        max_seq_len=max(T, ref_T) + 8,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        cond_dim=3,
    )
    model = RootDiffusionModel(**model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.2f}M  {model_config}")

    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    step = 0
    for epoch in range(args.epochs):
        model.train()
        t0, losses = time.time(), []
        for x, cond in loader:
            x = x.to(device)
            cond = cond.to(device)
            noise = torch.randn_like(x)
            ts = torch.randint(0, scheduler.config.num_train_timesteps,
                               (x.shape[0],), device=device)
            x_t = scheduler.add_noise(x, noise, ts)
            eps_pred = model(x_t, ts, cond)
            loss = torch.nn.functional.mse_loss(eps_pred, noise)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            step += 1
        mean_loss = sum(losses) / max(len(losses), 1)
        print(f"epoch {epoch:04d}  loss={mean_loss:.4f}  "
              f"dt={time.time()-t0:.1f}s  step={step}")

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.ckpt_dir, f"epoch_{epoch+1:04d}.pt")
            torch.save({
                "model": model.state_dict(),
                "model_config": model_config,
                "scheduler_config": dict(scheduler.config),
                "norm_stats": norm_stats,
                "data_meta": blob.get("meta", {}),
                "args": vars(args),
                "epoch": epoch + 1,
            }, ckpt_path)
            print(f"[ckpt] saved {ckpt_path}")


if __name__ == "__main__":
    main()
