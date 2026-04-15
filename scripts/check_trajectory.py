import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="legged_gym/resources/dataset/diffusion_train/amass_root.pt")
    parser.add_argument("--num_samples", type=int, default=3, help="可视化几条轨迹")
    args = parser.parse_args()

    # 1. 加载数据
    data = torch.load(args.data_path)
    trajectories = data["trajectories"]  # (N, T, 3)
    meta = data["meta"]
    
    print(f"Loaded {trajectories.shape[0]} trajectories.")
    print(f"Meta config: axis_perm={meta['axis_perm']}, axis_sign={meta['axis_sign']}")

    # 2. 随机抽取样本
    indices = np.random.choice(trajectories.shape[0], args.num_samples, replace=False)

    fig = plt.figure(figsize=(15, 5))
    
    for i, idx in enumerate(indices):
        traj = trajectories[idx].numpy()  # (T, 3)
        
        ax = fig.add_subplot(1, args.num_samples, i+1, projection='3d')
        
        x = traj[:, 0]
        y = traj[:, 1]
        z = traj[:, 2]  # 在 G1 坐标系下，这应该是高度

        # 绘制轨迹连线
        ax.plot(x, y, z, label='Trajectory', color='blue', alpha=0.7)
        # 散点表示时间步，颜色由浅入深
        ax.scatter(x, y, z, c=np.arange(len(x)), cmap='viridis', s=10, alpha=0.5)
        
        # 明确标记起点 (绿色) 和终点 (红色)
        ax.scatter(x[0], y[0], z[0], color='green', s=100, marker='*', label='Start')
        ax.scatter(x[-1], y[-1], z[-1], color='red', s=100, marker='X', label='End')

        ax.set_xlabel('X (Forward/Back)')
        ax.set_ylabel('Y (Left/Right)')
        ax.set_zlabel('Z (Up/Down)')
        ax.set_title(f"Sample {idx}")
        ax.legend()

        # 强制 3D 坐标轴比例保持一致，防止视觉变形导致误判
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()