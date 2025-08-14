import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from tqdm import tqdm
from dataload import get_dataloader
from trajectory_model import LSTMModel
from test_utils import autoregressive_predict
import pandas as pd

def load_min_max_from_csv(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    min_vals = df["min"].to_numpy()
    max_vals = df["max"].to_numpy()
    return min_vals, max_vals

def denormalize_velocity(norm_vel, min_vals, max_vals):
    return norm_vel * (max_vals - min_vals) + min_vals

def velocity_to_position(velocities, dt=0.1):
    positions = [np.array([0.0, 0.0, 0.0])]
    for v in velocities:
        v = np.array(v)
        new_pos = positions[-1] + v * dt
        positions.append(new_pos)
    return np.array(positions[1:])

# Parameters
input_size = 3
hidden_size = 128
num_layers = 2
output_size = 3
num_epochs = 50
lr = 1e-4
stride = 60
seq_len = 120
pred_len = 400
model_path = "trained_model_3.pt"
base_path = "15"
denorm_path = "denorm.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

min_vals, max_vals = load_min_max_from_csv(denorm_path)

_, _, test_loader = get_dataloader(base_path=base_path, seq_len=seq_len, stride=stride)

model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
model.to(device)
model.eval()

idx = 28    # one sample in the first batch (batch_size=50)
print("Starting Test...")
with torch.no_grad():
    for x_test, y_test in test_loader:
        input_seq = x_test[idx]
        gt_vel_norm = y_test[idx].cpu().numpy()
        pred_vel_norm = autoregressive_predict(model, input_seq, pred_len=pred_len, device=device)
        break

gt_vel = denormalize_velocity(gt_vel_norm, min_vals, max_vals)
gt_pos = velocity_to_position(gt_vel)

pred_vel = denormalize_velocity(pred_vel_norm, min_vals, max_vals)
pred_pos = velocity_to_position(pred_vel)

"""
# === 3D Plot ===
print("3D grafik çiziliyor...")
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(gt_pos[:,0], gt_pos[:,1], gt_pos[:,2], label='Ground Truth')
ax.plot(pred_pos[:,0], pred_pos[:,1], pred_pos[:,2], label='Predicted (AR)', linestyle='dashed')
ax.legend()
ax.set_title("Trajectory Prediction")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
"""

print("Plotting 3D Figure...")
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], label='Ground Truth')
ax.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], label='Generated (AR)', linestyle='dashed')

ax.set_title("Trajectory Prediction")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Axis Configuration
x_limits = ax.get_xlim3d()
y_limits = ax.get_ylim3d()
z_limits = ax.get_zlim3d()

x_range = x_limits[1] - x_limits[0]
y_range = y_limits[1] - y_limits[0]
z_range = z_limits[1] - z_limits[0]
max_range = max(x_range, y_range, z_range)

x_middle = np.mean(x_limits)
y_middle = np.mean(y_limits)
z_middle = np.mean(z_limits)

ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])

ax.grid(True)
ax.legend()

plt.show()

# CSV Export
export_data = {
    "pred_x": pred_pos[:,0],
    "pred_y": pred_pos[:,1],
    "pred_z": pred_pos[:,2],
}
df_export = pd.DataFrame(export_data)
df_export.to_csv("predicted.csv", index=False)
print("Saved as CSV: predicted.csv")