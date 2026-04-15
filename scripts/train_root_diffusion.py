"""训练根轨迹扩散模型 (创新点① MVP C2).

输入:
    --data_path  AMASS 预处理产物 .pt, 字段 {trajectories: Tensor[N, T, 3], source_ids: List[str]}
    --ref_path   基准参考 (e.g. high_jump data.pt), 用于统计条件范围和长度对齐

输出:
    --ckpt_dir 下 epoch_xxx.pt, 含 model.state_dict() + scheduler config + meta

用法 (本地或远程 GPU):
    python scripts/train_root_diffusion.py \\
        --data_path resources/dataset/diffusion_train/amass_root.pt \\
        --ref_path  legged_gym/resources/dataset/g1_dof27_data/high_jump/output/data.pt \\
        --ckpt_dir  checkpoints/root_diffusion/high_jump \\
        --epochs 200 --batch_size 64 --lr 2e-4
"""

import argparse
import os
import time
import torch
from torch.utils.data import DataLoader, Dataset
from diffusers import DDPMScheduler

from legged_gym.diffusion.root_mdm import RootDiffusionModel


class RootTrajectoryDataset(Dataset):
    """(traj, Δp) 对; Δp 从 traj 末端-始端随机加抖动得到，作为弱监督条件。

    这给 MVP 一个可训条件通路。真实 Δp 语义在 generate 脚本里由用户给定。
    """

    def __init__(self, trajectories: torch.Tensor, cond_jitter: float = 0.1):
        # trajectories: (N, T, 3), assume 已做坐标/尺度对齐
        self.traj = trajectories.float()
        self.cond_jitter = cond_jitter

    def __len__(self):
        return self.traj.shape[0]

    def __getitem__(self, idx):
        x = self.traj[idx]  # (T, 3)
        delta_p = x[-1] - x[0]  # 总位移作为条件
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
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    blob = torch.load(args.data_path)
    trajectories = blob["trajectories"] if isinstance(blob, dict) else blob
    assert trajectories.dim() == 3 and trajectories.shape[-1] == 3, \
        f"trajectories must be (N, T, 3), got {tuple(trajectories.shape)}"
    N, T, _ = trajectories.shape
    print(f"[data] N={N}  T={T}")

    ref = torch.load(args.ref_path)
    ref_T = ref["base_position"].shape[0]
    if T != ref_T:
        print(f"[warn] training T={T} != reference T={ref_T}; generate 时会按 ref_T 截/插")

    dataset = RootTrajectoryDataset(trajectories)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True)

    model = RootDiffusionModel(
        traj_dim=3, max_seq_len=max(T, ref_T) + 8,
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads, cond_dim=3,
    ).to(device)
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
            ts = torch.randint(0, scheduler.config.num_train_timesteps, (x.shape[0],), device=device)
            x_t = scheduler.add_noise(x, noise, ts)
            eps_pred = model(x_t, ts, cond)
            loss = torch.nn.functional.mse_loss(eps_pred, noise)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            step += 1
        print(f"epoch {epoch:04d}  loss={sum(losses)/len(losses):.4f}  "
              f"dt={time.time()-t0:.1f}s  step={step}")

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.ckpt_dir, f"epoch_{epoch+1:04d}.pt")
            torch.save({
                "model": model.state_dict(),
                "model_config": dict(
                    traj_dim=3, max_seq_len=max(T, ref_T) + 8,
                    d_model=args.d_model, n_layers=args.n_layers,
                    n_heads=args.n_heads, cond_dim=3,
                ),
                "scheduler_config": dict(scheduler.config),
                "args": vars(args),
                "epoch": epoch + 1,
            }, ckpt_path)
            print(f"[ckpt] saved {ckpt_path}")


if __name__ == "__main__":
    main()
