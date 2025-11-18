import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.optim.lr_scheduler as lr_scheduler
import random
import sys
import time
from scipy.spatial.transform import Rotation
import numpy.linalg as la
import torch.autograd.functional as F_autograd
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import copy

# shooting time in seconds
shooting_time = 1

# Time separation between samples in seconds when we consider a time disconitnuity and samples missing
# has to be bigger than the avarage sampling rate of the dataset
t_separation = 0.051

# Samples prediction multiplier factor
# sampling_time = sampling_time_dataset * pred_samples_factor
pred_samples_factor = 1

# Loss
# just_last_sample = 1 - only the last sample of each chunk is used on the loss computation
just_last_sample = 0

# Tolerances to be used in the odeint solver
rtol_aux = 1e-5
atol_aux = 1e-7

# Show final plots
show = 0

# Scheduler
flag_scheduler = 0

# Loss hyperparameters for each state variable
hparam_pos = 0.9441
hparam_vel = 0.0376
hparam_quat = 0.0182
hparam_ang_v = 0.0002

# Weight for penalizing NN residual dynamics (regularization strength)
lambda_nn_reg = 0.0001
nn_residual_norms = []

# Dataset
folder_gimble = "full_system"
training_dataset = "fs_sim32shots_first"
# preffix = ""
# suffix = ""
preffix = "simnl_"
suffix = "_QM4"

# Defining time limits t1 and t2 to select a portion of the dataset
select_data = 0
t_select_data_initial = 0
t_select_data_final = 20

# Interval of samples to be removed before and after actuation changed, in seconds
t_u_flag = 0
t_u_before = 0.1
t_u_after = 0.1

# Optimizer and learning parameters
lr = 5e-5
num_epochs = 800
weight_decay = 1e-4

# NN model architecture
nn_model_id = 4
hidden_dim = 1024

# Early stopping
early_stop = 0
best_valloss_model = 1
patience = 2000
warmup_epochs = 0
#flag_validation = 0- validation set is separate
#flag_validation = 1- validation set is part of the training set, x% randomly selected
#flag_validation = 2- validation set is part of the training set, every x segments are validation
flag_validation = 0
#flag_validation = 0:
validation_dataset = "fs_sim10shots3"
# validation_dataset = "fs_sim50shots"
# validation_dataset = "fs_sim_first"
#flag_validation = 1: Determines what fraction (%) of eligible chunks are randomly chosen for validation
percentage_validation = 20
#flag_validation = 2: Determines how often a chunk goes to validation.
validation_chunks = 5
# Defines the minimum fraction of shooting_time a segment must have to be considered valid
percentage_consider_chunk = 0.9

# Options for the plots
plot_hparam_pos = 0.8127
plot_hparam_vel = 0.1236
plot_hparam_quat = 0.0616
plot_hparam_ang_v = 0.0021
datasets_vec_original = [training_dataset, 'fs_sim_rand3', 'fs_sim_rand4', 'fs_sim24', 'fs_sim30', 'fs_sim46', 'fs_sim10shots', 'fs_sim10shots2', 'fs_sim10shots5', 'fs_sim20shots2', 'fs_sim50shots', 'fs_sim50shots2', 'fs_sim100shots2'] # the training dataset
plot_select_data_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # no selection of data
plot_t_select_data_initial_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
plot_t_select_data_final_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
plot_t_u_flag_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # no guard
plot_t_u_before_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
plot_t_u_after_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Load initial physical parameters computed by SPD approach
ms_sim_A1M = np.load('./folder/spd_A1M.npy')
ms_sim_A1F = np.load('./folder/spd_A1F.npy')
ms_sim_c = np.load('./folder/spd_c.npy')
ms_sim_c = np.array(ms_sim_c)
ms_sim_c = ms_sim_c.reshape(3, 1)
ms_sim_J1 = np.load('./folder/spd_J1.npy')

# Produce the final file names for the desired system
datasets_vec = []
aux_file_path = preffix + training_dataset + suffix
file_path = "./data/" + folder_gimble + "/" + aux_file_path + ".txt"
aux_file_path = preffix + validation_dataset + suffix
validation_file_path = "./data/" + folder_gimble + "/" + aux_file_path + ".txt"
for k in range(len(datasets_vec_original)):
    aux = preffix + datasets_vec_original[k] + suffix
    datasets_vec.append(aux)


# Read samples
def read_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    timestamp_list = []
    pos_list = []
    vel_list = []
    ang_v_list = []
    quaternions_list = []
    actuation_list = []

    i = 0
    while i < len(lines):
        if lines[i].strip():  # If the line is not empty
            # 1) timestamp
            timestamp = float(lines[i].strip())
            timestamp_list.append(timestamp)
            i += 1
            
            # 2) position (x, y, z)
            pos = list(map(float, lines[i].strip().split()))
            pos_list.append(pos)
            i += 1
            
            # 3) velocity (vx, vy, vz)
            vel = list(map(float, lines[i].strip().split()))
            vel_list.append(vel)
            i += 1
            
            # 4) angular velocity (wx, wy, wz)
            ang_v = list(map(float, lines[i].strip().split()))
            ang_v_list.append(ang_v)
            i += 1
            
            # 5) quaternion (bi, cj, dk, a)
            quaternions = list(map(float, lines[i].strip().split()))
            quaternions_list.append(quaternions)
            i += 1
            
            # 6) actuation (u_tk[0] ... u_tk[5])
            actuation = list(map(float, lines[i].strip().split()))
            actuation_list.append(actuation)
            i += 1
        else:
            i += 1  # Skip empty line
    
    # Convert to torch tensors
    timestamp_tensor = torch.tensor(timestamp_list, dtype=torch.float32)
    pos_tensor = torch.tensor(pos_list, dtype=torch.float32)
    vel_tensor = torch.tensor(vel_list, dtype=torch.float32)
    ang_v_tensor = torch.tensor(ang_v_list, dtype=torch.float32)
    quaternions_tensor = torch.tensor(quaternions_list, dtype=torch.float32)
    actuation_tensor = torch.tensor(actuation_list, dtype=torch.float32)
    
    return timestamp_tensor, pos_tensor, vel_tensor, ang_v_tensor, quaternions_tensor, actuation_tensor

# Function to recover L, c, AM and AF
def recover_params(params):
    # Sizes of L, c, AM and AF based on their shapes
    L_size = 6
    c_size = 3
    AM_size = 18
    AF_size = 18

    L = params[:L_size]
    L_aux = L.view(-1)
    L = torch.zeros(3, 3, dtype=L.dtype, device=L.device)
    L[0, 0] = L_aux[0]
    L[1, 0] = L_aux[1]
    L[1, 1] = L_aux[2]
    L[2, 0] = L_aux[3]
    L[2, 1] = L_aux[4]
    L[2, 2] = L_aux[5]

    c = params[L_size:L_size + c_size].view(3, 1)
    A1M = params[L_size + c_size:L_size + c_size + AM_size].view(3, 6)
    A1F = params[L_size + c_size + AM_size:L_size + c_size + AM_size + AF_size].view(3, 6)

    return L, c, A1M, A1F

# Function to recover position, linear velocity, quaternion and angular velocity from the state
def recover_state(state):
    # Sizes of each of the state variables
    pos_size = 3
    vel_size = 3
    quat_size = 4
    ang_v_size = 3

    # Split state back into position, velocity, quat and ang_v
    pos = state[:pos_size].view(3, 1)
    vel = state[pos_size:pos_size + vel_size].view(3, 1)
    quat = state[pos_size + vel_size:pos_size + vel_size + quat_size].view(4, 1)
    ang_v = state[pos_size + vel_size + quat_size:pos_size + vel_size + quat_size + ang_v_size].view(3, 1)

    return pos, vel, quat, ang_v

# skew-symmetric operator
def S(v):
    """
            |   0    -v3     v2 |
    S(v) =  |  v3      0    -v1 |
            | -v2     v1      0 |
    """
    # Ensure v is a 1D tensor of size 3
    v_aux = v.view(-1)

    # Create a zeros tensor of shape (3, 3) with the same dtype and device as v
    skew_symmetric_matrix = torch.zeros(3, 3, dtype=v.dtype, device=v.device)

    # Fill the skew-symmetric matrix
    skew_symmetric_matrix[0, 1] = -v_aux[2]
    skew_symmetric_matrix[0, 2] = v_aux[1]
    skew_symmetric_matrix[1, 0] = v_aux[2]
    skew_symmetric_matrix[1, 2] = -v_aux[0]
    skew_symmetric_matrix[2, 0] = -v_aux[1]
    skew_symmetric_matrix[2, 1] = v_aux[0]

    return skew_symmetric_matrix

# omega operator
def Omega(v):
    """
                |  0      v3     -v2     v1 |
    Omega(v) =  | -v3      0      v1     v2 |
                |  v2    -v1      0      v3 |
                | -v1    -v2     -v3     0  |
    """
    # Ensure v is a 1D tensor of size 3
    v_aux = v.view(-1)

    # Create a zeros tensor of shape (3, 3) with the same dtype and device as v
    matrix = torch.zeros(4, 4, dtype=v.dtype, device=v.device)

    # Fill the skew-symmetric matrix
    matrix[0, 1] = v_aux[2]
    matrix[0, 2] = -v_aux[1]
    matrix[0, 3] = v_aux[0]
    matrix[1, 0] = -v_aux[2]
    matrix[1, 2] = v_aux[0]
    matrix[1, 3] = v_aux[1]
    matrix[2, 0] = v_aux[1]
    matrix[2, 1] = -v_aux[0]
    matrix[2, 3] = v_aux[2]
    matrix[3, 0] = -v_aux[0]
    matrix[3, 1] = -v_aux[1]
    matrix[3, 2] = -v_aux[2]

    return matrix

# Quaternion to rotation matrix
def quat_to_R(quat):
    """
    Construct a 3x3 rotation matrix R from a quaternion [x, y, z, w].
            | 1 - 2(y^2 + z^2)   2(xy - zw)         2(xz + yw)   |
    R(q) =  | 2(xy + zw)         1 - 2(x^2 + z^2)   2(yz - xw)   |
            | 2(xz - yw)         2(yz + xw)         1 - 2(x^2 + y^2) |
    Args:
        quat (torch.Tensor): Quaternion tensor of shape (4,)
                             ordered as (x, y, z, w).
    Returns:
        torch.Tensor: Rotation matrix of shape (3, 3).
    """
    # Ensure quat is a 1D tensor of size 4
    q = quat.view(-1)
    x_q, y_q, z_q, w_q = q

    # Create a zeros tensor of shape (3, 3) with the same dtype and device as quat
    R = torch.zeros(3, 3, dtype=quat.dtype, device=quat.device)

    # Fill the rotation matrix
    R[0, 0] = 1 - 2 * (y_q**2 + z_q**2)
    R[0, 1] = 2 * (x_q * y_q - z_q * w_q)
    R[0, 2] = 2 * (x_q * z_q + y_q * w_q)

    R[1, 0] = 2 * (x_q * y_q + z_q * w_q)
    R[1, 1] = 1 - 2 * (x_q**2 + z_q**2)
    R[1, 2] = 2 * (y_q * z_q - x_q * w_q)

    R[2, 0] = 2 * (x_q * z_q - y_q * w_q)
    R[2, 1] = 2 * (y_q * z_q + x_q * w_q)
    R[2, 2] = 1 - 2 * (x_q**2 + y_q**2)

    return R

# Define the dynamics of the system
def f_mb(t, x, params, u):
    L, c, A1M, A1F = recover_params(params)
    _, vel, quat, ang_v = recover_state(x)

    # Transform actuation vector in 6x1 instead of 1x6
    u = u.view(-1, 1)

    # Compute R using quat
    R = quat_to_R(quat)

    # Compute the quaternions derivative
    dx3dt = 0.5 * torch.matmul(Omega(ang_v), quat)
    # dx3dt = 0.5 * torch.matmul(Operator(quat), ang_v)

    # Compute the angular acceleration derivative
    M = torch.matmul(A1M, u)
    L_gram = torch.inverse(torch.matmul(L, L.transpose(0, 1)))
    dx4dt = torch.matmul(L_gram, (M - torch.matmul(S(ang_v), torch.matmul(L, torch.matmul(L.transpose(0, 1), ang_v)))))

    # Compute the position derivative
    dx1dt = vel

    # Compute the velocity derivative
    F = torch.matmul(R, torch.matmul(A1F, u))
    dx2dt = F + torch.matmul(R, torch.matmul(S(c), dx4dt)) - torch.matmul(R, torch.matmul(torch.matmul(S(ang_v), S(ang_v)), c))

    # Concatenate all derivatives into a single vector  
    dxdt = torch.cat([dx1dt.view(-1), dx2dt.view(-1), dx3dt.view(-1), dx4dt.view(-1)])

    return dxdt

# NN structures for the model dynamcics
class DynamicsMLP1(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=64, output_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x_in):
        return self.net(x_in) # shape: (7,)

class DynamicsMLP2(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=128, output_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(hidden_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim / 2), output_dim),
        )
    def forward(self, x_in):
        out = self.net(x_in)
        return out  # shape: (7,)

class DynamicsMLP3(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=128, output_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x_in):
        out = self.net(x_in)
        return out  # shape: (7,)
    
class DynamicsMLP4(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=128, output_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x_in):
        out = self.net(x_in)
        return out  # shape: (7,)
    
class DynamicsMLP5(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=128, output_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x_in):
        out = self.net(x_in)
        return out  # shape: (7,)

class DynamicsMLP6(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=128, output_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x_in):
        out = self.net(x_in)
        return out  # shape: (7,)

# Define the loss function L
def loss_fn(state, true_pos, true_vel, true_quat, true_ang_v, hparam_pos, hparam_vel, hparam_quat, hparam_ang_v):
    # Get the predicted quaternion from the state
    pred_pos, pred_vel, pred_quat, pred_ang_v = recover_state(state)

    # Dot Product Loss for the quaternions
    pred_quat = pred_quat / torch.norm(pred_quat)
    true_quat = true_quat / torch.norm(true_quat)
    dot_product = torch.dot(pred_quat.view(-1), true_quat)
    # Loss is 1 - dot_product squared
    quat_loss = 1 -1 * (dot_product) ** 2
    
    # Mean Squared Difference for the remaining state variable losses
    pos_loss = F.mse_loss(pred_pos.view(-1), true_pos)
    vel_loss = F.mse_loss(pred_vel.view(-1), true_vel)
    ang_v_loss = F.mse_loss(pred_ang_v.view(-1), true_ang_v)

    # Total Loss
    loss = hparam_pos*pos_loss + hparam_vel*vel_loss + hparam_quat*quat_loss + hparam_ang_v*ang_v_loss

    return loss

# Function to get the shooting intervals
def get_shooting_intervals(t_aux, shooting_time, t_separation):
    """
    1. Splits t_aux at time discontinuities (where dt > t_separation).
    2. Then splits each continuous chunk into subchunks of shooting_time duration,
       measured from the start of that chunk (not absolute time).
    Returns a list of (start_idx, end_idx) tuples.
    """
    intervals = []
    t_aux = np.array(t_aux)

    # Step 1: Find discontinuities
    diffs = np.diff(t_aux)
    split_points = np.where(diffs > t_separation)[0]
    segment_indices = [0] + (split_points + 1).tolist() + [len(t_aux)]

    # Step 2: For each continuous chunk
    for i in range(len(segment_indices) - 1):
        start = segment_indices[i]
        end = segment_indices[i + 1]
        chunk_indices = np.arange(start, end)
        chunk_times = t_aux[chunk_indices]
        
        # Measure time relative to the start of the chunk
        relative_times = chunk_times - chunk_times[0]

        sub_start = 0
        while sub_start < len(chunk_indices):
            sub_start_time = relative_times[sub_start]
            sub_end_time = sub_start_time + shooting_time

            # Find all points in this subchunk
            sub_end = sub_start
            while (sub_end < len(relative_times) and 
                   relative_times[sub_end] < sub_end_time):
                sub_end += 1

            # If we have valid points, save the interval
            if sub_end > sub_start:
                start_idx = chunk_indices[sub_start]
                end_idx = chunk_indices[sub_end - 1]
                intervals.append((start_idx, end_idx))

            # Move to next subchunk
            sub_start = sub_end

    return intervals

# Function to find unique actuation segments
def find_unique_actuation_segments(actuation, start_sample_chunk, end_sample_chunk):
    """
    Finds unique actuation values and their index ranges within the specified range.

    Args:
        actuation (torch.Tensor): Tensor of shape (num_samples, actuation_dim) containing actuation values.
        start_sample_chunk (int): Start index of the chunk.
        end_sample_chunk (int): End index of the chunk (inclusive).

    Returns:
        index_u_chunk_groups (torch.Tensor): Tensor of shape (n, 2) containing start and end indices for each unique actuation segment.
        u_chunk_groups (torch.Tensor): Tensor of shape (n, actuation_dim) containing unique actuation values.
    """

    # Extract the relevant portion of the actuation tensor
    actuation_chunk = actuation[start_sample_chunk:end_sample_chunk + 1]

    # Initialize lists to store results
    index_u_chunk_groups = []
    u_chunk_groups = []

    # Track start index of the current unique segment
    prev_index = start_sample_chunk

    # Iterate through actuation_chunk to find constant segments
    for i in range(1, len(actuation_chunk)):
        if not torch.equal(actuation_chunk[i], actuation_chunk[i - 1]):  # Detect change in actuation
            index_u_chunk_groups.append([prev_index, start_sample_chunk + i - 1])
            u_chunk_groups.append(actuation_chunk[i - 1])
            prev_index = start_sample_chunk + i

    # Append the last segment
    index_u_chunk_groups.append([prev_index, end_sample_chunk])
    u_chunk_groups.append(actuation_chunk[-1])

    # Convert lists to tensors
    u_chunk_groups = torch.stack(u_chunk_groups)

    return index_u_chunk_groups, u_chunk_groups

# Multiple shooting function
def shooting(nn_model, t, t_np, positions, velocities, ang_v, quaternions, actuation, just_last_sample, rtol_aux, atol_aux, shooting_time, hparam_pos, hparam_vel, hparam_quat, hparam_ang_v, t_separation, physical_params, lambda_nn_reg):
   
    # Define the dynamics of the system
    def f_wrapper(t, x, physical_params, nn_model, u):
        # Prepare the input for the neural network
        pos, vel, quat, ang_v= recover_state(x)
        x_q, y_q, z_q, w_q = quat
        R = torch.tensor([
            [1 - 2*(y_q**2 + z_q**2),   2*(x_q*y_q - z_q*w_q),     2*(x_q*z_q + y_q*w_q)],
            [2*(x_q*y_q + z_q*w_q),     1 - 2*(x_q**2 + z_q**2),   2*(y_q*z_q - x_q*w_q)],
            [2*(x_q*z_q - y_q*w_q),     2*(y_q*z_q + x_q*w_q),     1 - 2*(x_q**2 + y_q**2)]
        ], dtype=x.dtype, device=x.device)
        R_flat = R.view(-1)
        x_in = torch.cat([pos.view(-1), vel.view(-1), R_flat, ang_v.view(-1), u.view(-1)], dim=0)  # single input vector

        # From kinematics MB
        pos_d = vel
        quat_d = 0.5 * torch.matmul(Omega(ang_v), quat)

        # Hybrid approach
        vel_d_residual = nn_model(x_in)[0:3]               
        vel_d_mb = f_mb(t, x, physical_params, u)[3:6]   
        vel_d = vel_d_mb + vel_d_residual

        angv_d_residual = nn_model(x_in)[3:6]                
        angv_d_mb = f_mb(t, x, physical_params, u)[10:]   
        angv_d = angv_d_mb + angv_d_residual

        dxdt = torch.cat([pos_d.view(-1), vel_d.view(-1), quat_d.view(-1), angv_d.view(-1)])

        # Save the residual dynamics and the MB dynamics for loss penalization
        nn_residual_norms.append(torch.norm(nn_model(x_in)))

        return dxdt
    
    time_odeint = []
    pred_positions = []
    pred_velocities = []
    pred_quaternions = []
    pred_ang_v = []
    total_loss = torch.tensor(0.0, requires_grad=True)

    # Get start and end indices for the shooting intervals
    shooting_intervals = get_shooting_intervals(t_np, shooting_time, t_separation)

    # Pass through each chunk
    for chunk in range(len(shooting_intervals)):

        # Get the start sample index for the current chunk
        start_sample_chunk, end_sample_chunk = shooting_intervals[chunk]

        # Find the unique actuation values and respective indices for the current chunk
        index_u_chunk_groups, u_chunk_groups = find_unique_actuation_segments(actuation, start_sample_chunk, end_sample_chunk)

        # Pass through each segment of constant actuation
        for i in range(len(index_u_chunk_groups)):

            # Get the unique actuation value for the current segment and the index range
            start_sample_chunk_i = index_u_chunk_groups[i][0]
            end_sample_chunk_i = index_u_chunk_groups[i][1]
            actuation_value = u_chunk_groups[i]

            # Reset the state for that actuation segment of the chunk
            if i == 0: #If it is the first segment of the chunk, we reset the state
                state = torch.cat([positions[start_sample_chunk_i], velocities[start_sample_chunk_i], quaternions[start_sample_chunk_i], ang_v[start_sample_chunk_i]])
            else: #If it is not the first segment of the chunk, we keep the last state of the previous segment and update the actuation value
                state = prev_last_state

            # Set end index of next prediction
            if i == (len(index_u_chunk_groups)-1): 
                end_sample_chunk_i_aux = end_sample_chunk_i
            else: # If it is not the last segment of the chunk, we set the end index to the next segment start index to predict next segment initial state
                end_sample_chunk_i_aux = end_sample_chunk_i + 1

            # Solve the differential equation, predicting the next state
            start_odeint_time = time.time()
            predicted_next_states_shoot_i = odeint(
                lambda t, x: f_wrapper(t, x, physical_params, nn_model, actuation_value),
                # f_wrapper,
                state,
                t[start_sample_chunk_i:end_sample_chunk_i_aux+1].float(), # t from index start_sample_chunk_i to end_sample_chunk_i_aux (needs the +1)
                rtol=rtol_aux,  # Relative tolerance
                atol=atol_aux,  # Absolute tolerance
                method='dopri5'
            )
            end_odeint_time = time.time()
            time_odeint.append(end_odeint_time - start_odeint_time)

            # Add to the loss the penalization of th residual NN learning
            if len(nn_residual_norms) > 0:
                nn_reg_term = torch.stack(nn_residual_norms).sum()
                total_loss = total_loss + lambda_nn_reg * nn_reg_term
                nn_residual_norms.clear()

            # If it is not the last segment of the chunk, we need to set the last state of the segment as the initial state for the next segment
            if i != (len(index_u_chunk_groups)-1): 
                prev_last_state = predicted_next_states_shoot_i[-1] # Store the last state of the segment to be used as initial state for the next segment
                predicted_next_states_shoot_i = predicted_next_states_shoot_i[:-1] # Remove the last state as it is the initial state of the next segment

            # Concatenate the predicted states of the segments for that chunk
            if i == 0:
                predicted_next_states_shoot = predicted_next_states_shoot_i
            else:
                predicted_next_states_shoot = torch.cat([predicted_next_states_shoot, predicted_next_states_shoot_i], dim=0)
            
        # Store the predicted state
        pred_np = predicted_next_states_shoot.detach().cpu().numpy()
        for i in range(predicted_next_states_shoot.shape[0]):
            pred_positions.append(pred_np[i,:3])
            pred_velocities.append(pred_np[i,3:6])
            pred_quaternions.append(pred_np[i,6:10])
            pred_ang_v.append(pred_np[i,10:13])

        # Loss computation
        if just_last_sample == 1: # only with the last predicted state of the shoot
            loss_value = loss_fn(predicted_next_states_shoot[-1], positions[end_sample_chunk], velocities[end_sample_chunk], quaternions[end_sample_chunk], ang_v[end_sample_chunk], hparam_pos, hparam_vel, hparam_quat, hparam_ang_v)
            total_loss = total_loss + loss_value
        else: # with all the predicted states of the shoot
            for i in range(predicted_next_states_shoot.shape[0]):
                loss_value = loss_fn(predicted_next_states_shoot[i], positions[start_sample_chunk + i], velocities[start_sample_chunk + i], quaternions[start_sample_chunk + i], ang_v[start_sample_chunk + i], hparam_pos, hparam_vel, hparam_quat, hparam_ang_v)
                total_loss = total_loss + loss_value

    return total_loss, torch.tensor(pred_positions), torch.tensor(pred_velocities), torch.tensor(pred_quaternions), torch.tensor(pred_ang_v), sum(time_odeint)

def filter_tensors_on_actuation_change(t, positions, velocities, ang_v, quaternions, actuation, t_u_before, t_u_after):
    # Convert time tensor to numpy for easier manipulation
    t_np = t.numpy() if isinstance(t, torch.Tensor) else t
    act_np = actuation.numpy() if isinstance(actuation, torch.Tensor) else actuation
    
    # Find where the actuation changes (row-wise)
    changes = np.any(np.diff(act_np, axis=0) != 0, axis=1)
    change_indices = np.where(changes)[0] + 1  # because diff is offset by 1
    
    # Mark indices to keep (start with all True)
    keep_mask = np.ones_like(t_np, dtype=bool)

    for idx in change_indices:
        t_center = t_np[idx]
        # Remove samples outside the time window around the change
        outside_window = (t_np < (t_center - t_u_before)) | (t_np > (t_center + t_u_after))
        keep_mask &= outside_window  # only keep samples that are outside all windows
    
    # Apply mask to all tensors
    t_new = t[keep_mask]
    positions_new = positions[keep_mask]
    velocities_new = velocities[keep_mask]
    ang_v_new = ang_v[keep_mask]
    quaternions_new = quaternions[keep_mask]
    actuation_new = actuation[keep_mask]

    return t_new, positions_new, velocities_new, ang_v_new, quaternions_new, actuation_new

def split_val_train_chunks(t, positions, velocities, ang_v, quaternions, actuation, t_separation, shooting_time, percentage_validation, percentage_consider_chunk, flag_validation, validation_chunks):
    segments = []
    start_idx = 0

    # Step 1: Identify valid segments
    i = 1
    while i < len(t):
        time_gap = t[i] - t[i - 1]
        segment_duration = t[i - 1] - t[start_idx]

        # Case 1: Large time gap and segment has enough duration
        if time_gap > t_separation:
            if segment_duration >= percentage_consider_chunk * shooting_time:
                segments.append((start_idx, i))
            start_idx = i
            i += 1
            continue

        # Case 2: Segment has reached or exceeded shooting_time
        full_duration = t[i] - t[start_idx]
        if full_duration >= shooting_time:
            segments.append((start_idx, i))  # i is NOT included
            start_idx = i  # Start next segment from i
            i += 1
            continue

        i += 1

    # Check the last segment
    if start_idx < len(t):
        duration = t[-1] - t[start_idx]
        if duration >= percentage_consider_chunk * shooting_time:
            segments.append((start_idx, len(t)))

    # Pick randomly a percentage of the chunks as validation dataset
    if flag_validation == 1:
        # Randomly pick percentage_validation% of *all* segments for validation
        num_val = int(round(len(segments) * percentage_validation / 100.0))
        chosen_val_indices = set(random.sample(range(len(segments)), num_val))

    # Pick every validation_chunks-th segment for validation
    elif flag_validation == 2:
        # elect every validation_chunks-th segment for validation
        chosen_val_indices = set(i for i in range(len(segments)) if (i + 1) % validation_chunks == 0)

    # Split data
    t_train, positions_train, velocities_train, ang_v_train, quaternions_train, actuation_train = [], [], [], [], [], []
    t_val, positions_val, velocities_val, ang_v_val, quaternions_val, actuation_val = [], [], [], [], [], []

    for i, (start, end) in enumerate(segments):
        target_tensors = (
            (t_val, positions_val, velocities_val, ang_v_val, quaternions_val, actuation_val)
            if i in chosen_val_indices else
            (t_train, positions_train, velocities_train, ang_v_train, quaternions_train, actuation_train)
        )
        target_tensors[0].append(t[start:end])
        target_tensors[1].append(positions[start:end])
        target_tensors[2].append(velocities[start:end])
        target_tensors[3].append(ang_v[start:end])
        target_tensors[4].append(quaternions[start:end])
        target_tensors[5].append(actuation[start:end])

    # Concatenate into final tensors
    t_train = torch.cat(t_train) if t_train else torch.tensor([])
    positions_train = torch.cat(positions_train) if positions_train else torch.tensor([])
    velocities_train = torch.cat(velocities_train) if velocities_train else torch.tensor([])
    ang_v_train = torch.cat(ang_v_train) if ang_v_train else torch.tensor([])
    quaternions_train = torch.cat(quaternions_train) if quaternions_train else torch.tensor([])
    actuation_train = torch.cat(actuation_train) if actuation_train else torch.tensor([])

    t_val = torch.cat(t_val) if t_val else torch.tensor([])
    positions_val = torch.cat(positions_val) if positions_val else torch.tensor([])
    velocities_val = torch.cat(velocities_val) if velocities_val else torch.tensor([])
    ang_v_val = torch.cat(ang_v_val) if ang_v_val else torch.tensor([])
    quaternions_val = torch.cat(quaternions_val) if quaternions_val else torch.tensor([])
    actuation_val = torch.cat(actuation_val) if actuation_val else torch.tensor([])

    return {
        'segments_found': segments,
        'chosen_validation_indices': chosen_val_indices,
        't_train': t_train,
        'positions_train': positions_train,
        'velocities_train': velocities_train,
        'ang_v_train': ang_v_train,
        'quaternions_train': quaternions_train,
        'actuation_train': actuation_train,
        't_val': t_val,
        'positions_val': positions_val,
        'velocities_val': velocities_val,
        'ang_v_val': ang_v_val,
        'quaternions_val': quaternions_val,
        'actuation_val': actuation_val
    }


# Function to change the final vectors and avoid connecting points to far away in the final graphs
def break_plot_on_gaps(t, y, max_gap):
    if torch.is_tensor(t):
        t = t.detach().cpu().numpy()
    else:
        t = np.asarray(t)
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()
    else:
        y = np.asarray(y)
    dt = np.diff(t)

    # Find indices where the time gap is too large
    break_indices = np.where(dt > max_gap)[0]

    # Prepare lists to hold the new t and y with NaNs inserted
    t_new = []
    y_new = []

    for i in range(len(t)):
        t_new.append(t[i])
        y_new.append(y[i])
        if i in break_indices:
            # Insert NaN to break the plot
            t_new.append(np.nan)
            y_new.append(np.nan)

    return np.array(t_new), np.array(y_new)

# Function to convert quaternions to Euler angles (phi, theta, psi)
def quaternion_to_euler_angles(quaternions):
    # Extract quaternion components
    b = quaternions[:, 0]  # i (x component)
    c = quaternions[:, 1]  # j (y component)
    d = quaternions[:, 2]  # k (z component)
    a = quaternions[:, 3]  # real part (w component)
    
    # Compute Euler angles using standard formulas
    phi = torch.atan2(2 * (a * b + c * d), 1 - 2 * (b**2 + c**2))
    theta = torch.asin(2 * (a * c - d * b))
    psi = torch.atan2(2 * (a * d + b * c), 1 - 2 * (c**2 + d**2))

    # Convert to list of tuples
    euler_angles = [(phi[i].item(), theta[i].item(), psi[i].item()) for i in range(quaternions.shape[0])]
    
    return euler_angles

# Function to compute RMSE
def compute_rmse(pred_angles, true_angles):
    
    if torch.is_tensor(pred_angles):
        pred_angles = pred_angles.detach().cpu().numpy()
    if torch.is_tensor(true_angles):
        true_angles = true_angles.detach().cpu().numpy()

    pred_angles = np.array(pred_angles)
    true_angles = np.array(true_angles)

    # Mask where neither value is NaN
    mask = ~np.isnan(pred_angles) & ~np.isnan(true_angles)
    pred_angles = pred_angles[mask]
    true_angles = true_angles[mask]

    # Compute squared differences
    squared_errors = (pred_angles - true_angles) ** 2

    # Compute mean squared error for each angle component
    mse = np.mean(squared_errors, axis=0)

    # Compute RMSE (take square root of MSE)
    rmse = np.sqrt(mse)

    return rmse

# Function to select a range of data based on timestamps
def select_data_range(t, positions, velocities, ang_v, quaternions, actuation, t_initial, t_final):
    # Ensure t is sorted
    if not (t[1:] >= t[:-1]).all():
        raise ValueError("t must be sorted in ascending order")

    # Index for x: largest t[i] <= x, or 0 if none
    i_candidates = (t <= t_initial).nonzero(as_tuple=True)[0]
    i = i_candidates[-1].item() if len(i_candidates) > 0 else 0

    # Index for y: largest t[j] <= y, or 0 if none
    j_candidates = (t <= t_final).nonzero(as_tuple=True)[0]
    j = j_candidates[-1].item() if len(j_candidates) > 0 else 0

    if i > j:
        raise ValueError("x must be less than or equal to y")

    # Slice the tensors
    t_selected = t[i:j+1]
    positions_selected = positions[i:j+1]
    velocities_selected = velocities[i:j+1]
    ang_v_selected = ang_v[i:j+1]
    quaternions_selected = quaternions[i:j+1]
    actuation_selected = actuation[i:j+1]

    return t_selected, positions_selected, velocities_selected, ang_v_selected, quaternions_selected, actuation_selected

# Function to plots shooting results, compute RMSE values and store them
def plot_shooting_results_rmse(datasets_vec, plot_select_data_vec, plot_t_select_data_initial_vec, plot_t_select_data_final_vec, plot_t_u_flag_vec, plot_t_u_before_vec, plot_t_u_after_vec, nn_model, hparam_pos, hparam_vel, hparam_quat, hparam_ang_v, pred_samples_factor, shooting_time, just_last_sample, rtol_aux, atol_aux, file, directory1, directory2, directory3, num_epochs, lr, optimizer, flag_scheduler, str_t_u_flag, folder_gimble, physical_params, lambda_nn_reg):

    mean_test_pos_rmse_vec = []
    mean_test_vel_rmse_vec = []
    mean_test_att_rmse_vec = []
    mean_test_ang_v_rmse_vec = []

    for i in range(len(datasets_vec)):

        # Write this shooting information in the results file
        if plot_select_data_vec[i] == 1:
            information_str = "shooting results tested on " + datasets_vec[i] + " " + str(plot_t_select_data_initial_vec[i]) + "-" + str(plot_t_select_data_final_vec[i]) + "s, with "
        else:
            information_str = "shooting results tested on " + datasets_vec[i] + ", with "
        if plot_t_u_flag_vec[i] == 1:
            information_str = information_str + "guard of " + str(plot_t_u_before_vec[i]) + "s before and " + str(plot_t_u_after_vec[i]) + "s after:"
        else:
            information_str = information_str + "no guard:"
        file.write(information_str + "\n")

        # Building the dataset file path
        file_path = datasets_vec[i]
        file_path = './data/' + folder_gimble + "/" + datasets_vec[i] + '.txt'

        if plot_t_u_flag_vec[i]:
            plot_str_t_u_flag = '_ptu'+str(plot_t_u_flag_vec[i])+'b'+str(plot_t_u_before_vec[i])+'a'+str(plot_t_u_after_vec[i])
        else:
            plot_str_t_u_flag = '_ptu'+str(plot_t_u_flag_vec[i])

        # Read sample data
        t, positions, velocities, ang_v, quaternions, actuation = read_data(file_path)

        # Setting time to start on zero
        t = t - t[0]

        # Selecting specific portions of the dataset with the timestamps t1 and t2
        if plot_select_data_vec[i] == 1:
            t, positions, velocities, ang_v, quaternions, actuation = select_data_range(t, positions, velocities, ang_v, quaternions, actuation, t_select_data_initial, t_select_data_final)

        # Only consider samples, to then be predicted, according to the pred_samples_factor
        if pred_samples_factor > 0 and isinstance(pred_samples_factor, int):
            t = t[::pred_samples_factor]
            positions = positions[::pred_samples_factor]
            velocities = velocities[::pred_samples_factor]
            ang_v = ang_v[::pred_samples_factor]
            quaternions = quaternions[::pred_samples_factor]
            actuation = actuation[::pred_samples_factor]
        else:
            print("\nerror pred_samples_factor it's not an integer.")
            sys.exit()

        # Remove true samples before and after actuation changes to avoid errors in the predictions and learning
        if plot_t_u_flag_vec[i]:
            t, positions, velocities, ang_v, quaternions, actuation = filter_tensors_on_actuation_change(t, positions, velocities, ang_v, quaternions, actuation, t_u_before, t_u_after)
        t_np = t.detach().cpu().numpy()

        # Get dataset size
        data_size = t.size()[0]

        # Move the model to the appropriate device (GPU or CPU) and tensor, to speed up the training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nn_model.to(device)
        # Move all tensors to device
        t = t.to(device)
        positions = positions.to(device)
        velocities = velocities.to(device)
        ang_v = ang_v.to(device)
        quaternions = quaternions.to(device)
        actuation = actuation.to(device)

        # Get the true Euler angles
        euler_angles = quaternion_to_euler_angles(quaternions)
        phi_angles = [item[0] for item in euler_angles]
        theta_angles = [item[1] for item in euler_angles]
        psi_angles = [item[2] for item in euler_angles]

        # Shooting
        loss,pred_positions,pred_velocities,pred_quaternions,pred_ang_v,_ = shooting(nn_model, t, t_np, positions, velocities, ang_v, quaternions, actuation, just_last_sample, rtol_aux, atol_aux, shooting_time, hparam_pos, hparam_vel, hparam_quat, hparam_ang_v, t_separation, physical_params, lambda_nn_reg)
        mean_loss = (loss.item())/data_size
        file.write(f"Total loss (hparam_pos={hparam_pos},hparam_vel={hparam_vel},hparam_quat={hparam_quat},hparam_ang_v={hparam_ang_v}): {loss.item()}\n")
        file.write(f"Mean Total loss (hparam_pos={hparam_pos},hparam_vel={hparam_vel},hparam_quat={hparam_quat},hparam_ang_v={hparam_ang_v}): {mean_loss}\n")

        # Get the predicted quaternions norms
        pred_quaternions_norms = []
        for j in range(len(pred_quaternions)):
            pred_quaternions_norms.append(pred_quaternions[j].norm().item())

        # Get the predicted attitudes in Euler angles
        pred_euler_angles = quaternion_to_euler_angles(pred_quaternions)
        pred_phi_angles = [item[0] for item in pred_euler_angles]
        pred_theta_angles = [item[1] for item in pred_euler_angles]
        pred_psi_angles = [item[2] for item in pred_euler_angles]

        # Defining directories to save the results
        aux_solver = optimizer.__class__.__name__ + "_lr=" + str(lr) + "_"+str(num_epochs)+"epochs_sch="+str(flag_scheduler)+"_"
        if plot_select_data_vec[i] == 1:
            bag_name = datasets_vec[i] + "_" + str(plot_t_select_data_initial_vec[i]) + "-" + str(plot_t_select_data_final_vec[i]) + "s"
        else:
            bag_name = datasets_vec[i]
        file_path1 = bag_name + '_phi_' + aux_solver + 'st=' +  str(shooting_time) + '_taf=' + str(pred_samples_factor) +str_t_u_flag +plot_str_t_u_flag
        file_path2 = bag_name + '_theta_' + aux_solver + 'st=' +  str(shooting_time) + '_taf=' + str(pred_samples_factor) +str_t_u_flag +plot_str_t_u_flag
        file_path3 = bag_name + '_psi_' + aux_solver + 'st=' +  str(shooting_time) + '_taf=' + str(pred_samples_factor) +str_t_u_flag +plot_str_t_u_flag
        file_path4 = bag_name + '_ang_v_x_' + aux_solver + 'st=' +  str(shooting_time) + '_taf=' + str(pred_samples_factor) +str_t_u_flag +plot_str_t_u_flag
        file_path5 = bag_name + '_ang_v_y_' + aux_solver + 'st=' +  str(shooting_time) + '_taf=' + str(pred_samples_factor) +str_t_u_flag +plot_str_t_u_flag
        file_path6 = bag_name + '_ang_v_z_' + aux_solver + 'st=' +  str(shooting_time) + '_taf=' + str(pred_samples_factor) +str_t_u_flag +plot_str_t_u_flag
        file_path7 = bag_name + '_pos_v_x_' + aux_solver + 'st=' +  str(shooting_time) + '_taf=' + str(pred_samples_factor) +str_t_u_flag +plot_str_t_u_flag
        file_path8 = bag_name + '_pos_v_y_' + aux_solver + 'st=' +  str(shooting_time) + '_taf=' + str(pred_samples_factor) +str_t_u_flag +plot_str_t_u_flag
        file_path9 = bag_name + '_pos_v_z_' + aux_solver + 'st=' +  str(shooting_time) + '_taf=' + str(pred_samples_factor) +str_t_u_flag +plot_str_t_u_flag
        file_path10 = bag_name + '_vel_v_x_' + aux_solver + 'st=' +  str(shooting_time) + '_taf=' + str(pred_samples_factor) +str_t_u_flag +plot_str_t_u_flag
        file_path11 = bag_name + '_vel_v_y_' + aux_solver + 'st=' +  str(shooting_time) + '_taf=' + str(pred_samples_factor) +str_t_u_flag +plot_str_t_u_flag
        file_path12 = bag_name + '_vel_v_z_' + aux_solver + 'st=' +  str(shooting_time) + '_taf=' + str(pred_samples_factor) +str_t_u_flag +plot_str_t_u_flag
        file_path13 = bag_name + '_quat_norm_' + aux_solver + 'st=' +  str(shooting_time) + '_taf=' + str(pred_samples_factor) +str_t_u_flag +plot_str_t_u_flag

        # Correcting the final prediction vectors so that two far away points are not connected
        t_plot, phi_angles_plot = break_plot_on_gaps(t, phi_angles, max_gap=t_separation)
        t_plot, pred_phi_angles_plot = break_plot_on_gaps(t, pred_phi_angles, max_gap=t_separation)
        t_plot, theta_angles_plot = break_plot_on_gaps(t, theta_angles, max_gap=t_separation)
        t_plot, pred_theta_angles_plot = break_plot_on_gaps(t, pred_theta_angles, max_gap=t_separation)
        t_plot, psi_angles_plot = break_plot_on_gaps(t, psi_angles, max_gap=t_separation)
        t_plot, pred_psi_angles_plot = break_plot_on_gaps(t, pred_psi_angles, max_gap=t_separation)
        t_plot, ang_v_plot0 = break_plot_on_gaps(t, ang_v[:, 0], max_gap=t_separation)
        t_plot, pred_ang_v_plot0 = break_plot_on_gaps(t, pred_ang_v[:, 0], max_gap=t_separation)
        t_plot, ang_v_plot1 = break_plot_on_gaps(t, ang_v[:, 1], max_gap=t_separation)
        t_plot, pred_ang_v_plot1 = break_plot_on_gaps(t, pred_ang_v[:, 1], max_gap=t_separation)
        t_plot, ang_v_plot2 = break_plot_on_gaps(t, ang_v[:, 2], max_gap=t_separation)
        t_plot, pred_ang_v_plot2 = break_plot_on_gaps(t, pred_ang_v[:, 2], max_gap=t_separation)
        t_plot, pred_quaternions_norms_plot = break_plot_on_gaps(t, pred_quaternions_norms, max_gap=t_separation)
        t_plot, velocities_plot0 = break_plot_on_gaps(t, velocities[:, 0], max_gap=t_separation)
        t_plot, pred_velocities_plot0 = break_plot_on_gaps(t, pred_velocities[:, 0], max_gap=t_separation)
        t_plot, velocities_plot1 = break_plot_on_gaps(t, velocities[:, 1], max_gap=t_separation)
        t_plot, pred_velocities_plot1 = break_plot_on_gaps(t, pred_velocities[:, 1], max_gap=t_separation)
        t_plot, velocities_plot2 = break_plot_on_gaps(t, velocities[:, 2], max_gap=t_separation)
        t_plot, pred_velocities_plot2 = break_plot_on_gaps(t, pred_velocities[:, 2], max_gap=t_separation)
        t_plot, positions_plot0 = break_plot_on_gaps(t, positions[:, 0], max_gap=t_separation)
        t_plot, pred_positions_plot0 = break_plot_on_gaps(t, pred_positions[:, 0], max_gap=t_separation)
        t_plot, positions_plot1 = break_plot_on_gaps(t, positions[:, 1], max_gap=t_separation)
        t_plot, pred_positions_plot1 = break_plot_on_gaps(t, pred_positions[:, 1], max_gap=t_separation)
        t_plot, positions_plot2 = break_plot_on_gaps(t, positions[:, 2], max_gap=t_separation)
        t_plot, pred_positions_plot2 = break_plot_on_gaps(t, pred_positions[:, 2], max_gap=t_separation)

        def to_numpy(x):
            if torch.is_tensor(x):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        # --- Convert all potentially GPU tensors before plotting ---
        t = to_numpy(t)
        t_plot = to_numpy(t_plot)

        # Positions
        positions_plot0 = to_numpy(positions_plot0)
        positions_plot1 = to_numpy(positions_plot1)
        positions_plot2 = to_numpy(positions_plot2)
        pred_positions_plot0 = to_numpy(pred_positions_plot0)
        pred_positions_plot1 = to_numpy(pred_positions_plot1)
        pred_positions_plot2 = to_numpy(pred_positions_plot2)

        # Velocities
        velocities_plot0 = to_numpy(velocities_plot0)
        velocities_plot1 = to_numpy(velocities_plot1)
        velocities_plot2 = to_numpy(velocities_plot2)
        pred_velocities_plot0 = to_numpy(pred_velocities_plot0)
        pred_velocities_plot1 = to_numpy(pred_velocities_plot1)
        pred_velocities_plot2 = to_numpy(pred_velocities_plot2)

        # Angles
        phi_angles_plot = to_numpy(phi_angles_plot)
        theta_angles_plot = to_numpy(theta_angles_plot)
        psi_angles_plot = to_numpy(psi_angles_plot)
        pred_phi_angles_plot = to_numpy(pred_phi_angles_plot)
        pred_theta_angles_plot = to_numpy(pred_theta_angles_plot)
        pred_psi_angles_plot = to_numpy(pred_psi_angles_plot)

        # Angular velocities
        ang_v_plot0 = to_numpy(ang_v_plot0)
        ang_v_plot1 = to_numpy(ang_v_plot1)
        ang_v_plot2 = to_numpy(ang_v_plot2)
        pred_ang_v_plot0 = to_numpy(pred_ang_v_plot0)
        pred_ang_v_plot1 = to_numpy(pred_ang_v_plot1)
        pred_ang_v_plot2 = to_numpy(pred_ang_v_plot2)

        # Quaternions norms
        pred_quaternions_norms_plot = to_numpy(pred_quaternions_norms_plot)

        # Position shooting ------------------------------------------------------------------
        plt.plot(t_plot, positions_plot0, label=r'True $p_x$')
        plt.plot(t_plot, pred_positions_plot0, label=r'True $p_x$')
        # Add vertical lines at multiples of shooting_time
        for st in np.arange(t[0], max(t), shooting_time):
            plt.axvline(x=st, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Time [s]')
        plt.ylabel('X Position [m]')
        plt.legend()
        plt.savefig(os.path.join(directory1, file_path7)+'.png', bbox_inches='tight')
        plt.savefig(os.path.join(directory2, file_path7)+'.pdf', bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        # Saving data points
        file_points = directory3 + '/pos_x_shooting.txt'
        with open(file_points, "w") as fpoints:
            fpoints.write("x = [")
            fpoints.write(" ".join(str(x) for x in t_plot))
            fpoints.write("];\n")
            fpoints.write("true = [")
            fpoints.write(" ".join(str(y) for y in positions_plot0))
            fpoints.write("];\n")
            fpoints.write("pred = [")
            fpoints.write(" ".join(str(y) for y in pred_positions_plot0))
            fpoints.write("];\n")

        plt.plot(t_plot, positions_plot1, label=r'True $p_y$')
        plt.plot(t_plot, pred_positions_plot1, label=r'Pred $p_y$')
        # Add vertical lines at multiples of shooting_time
        for st in np.arange(t[0], max(t), shooting_time):
            plt.axvline(x=st, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Time [s]')
        plt.ylabel('Y Position [m]')
        plt.legend()
        plt.savefig(os.path.join(directory1, file_path8)+'.png', bbox_inches='tight')
        plt.savefig(os.path.join(directory2, file_path8)+'.pdf', bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        # Saving data points
        file_points = directory3 + '/pos_y_shooting.txt'
        with open(file_points, "w") as fpoints:
            fpoints.write("x = [")
            fpoints.write(" ".join(str(x) for x in t_plot))
            fpoints.write("];\n")
            fpoints.write("true = [")
            fpoints.write(" ".join(str(y) for y in positions_plot1))
            fpoints.write("];\n")
            fpoints.write("pred = [")
            fpoints.write(" ".join(str(y) for y in pred_positions_plot1))
            fpoints.write("];\n")

        plt.plot(t_plot, positions_plot2, label=r'True $p_z$')
        plt.plot(t_plot, pred_positions_plot2, label=r'Pred $p_z$')
        # Add vertical lines at multiples of shooting_time
        for st in np.arange(t[0], max(t), shooting_time):
            plt.axvline(x=st, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Time [s]')
        plt.ylabel('Z Position [m]')
        plt.legend()
        plt.savefig(os.path.join(directory1, file_path9)+'.png', bbox_inches='tight')
        plt.savefig(os.path.join(directory2, file_path9)+'.pdf', bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        # Saving data points
        file_points = directory3 + '/pos_z_shooting.txt'
        with open(file_points, "w") as fpoints:
            fpoints.write("x = [")
            fpoints.write(" ".join(str(x) for x in t_plot))
            fpoints.write("];\n")
            fpoints.write("true = [")
            fpoints.write(" ".join(str(y) for y in positions_plot2))
            fpoints.write("];\n")
            fpoints.write("pred = [")
            fpoints.write(" ".join(str(y) for y in pred_positions_plot2))
            fpoints.write("];\n")

        # Compute RMSE for each angular velocity component
        pos_x_rmse = compute_rmse(pred_positions[:, 0], positions[:, 0])
        pos_y_rmse = compute_rmse(pred_positions[:, 1], positions[:, 1])
        pos_z_rmse = compute_rmse(pred_positions[:, 2], positions[:, 2])
        mean_pos_rmse = np.mean([pos_x_rmse, pos_y_rmse, pos_z_rmse])
        file.write(f"pos x rmse: {pos_x_rmse}\n")
        file.write(f"pos y rmse: {pos_y_rmse}\n")
        file.write(f"pos z rmse: {pos_z_rmse}\n")

        # Linear velocity shooting ------------------------------------------------------------------
        plt.plot(t_plot, velocities_plot0, label=r'True $v_x$')
        plt.plot(t_plot, pred_velocities_plot0, label=r'Pred $v_x$')
        # Add vertical lines at multiples of shooting_time
        for st in np.arange(t[0], max(t), shooting_time):
            plt.axvline(x=st, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Time [s]')
        plt.ylabel('X Linear Velocity [m/s]')   
        plt.legend()
        plt.savefig(os.path.join(directory1, file_path10)+'.png', bbox_inches='tight')
        plt.savefig(os.path.join(directory2, file_path10)+'.pdf', bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        # Saving data points
        file_points = directory3 + '/vel_x_shooting.txt'
        with open(file_points, "w") as fpoints:
            fpoints.write("x = [")
            fpoints.write(" ".join(str(x) for x in t_plot))
            fpoints.write("];\n")
            fpoints.write("true = [")
            fpoints.write(" ".join(str(y) for y in velocities_plot0))
            fpoints.write("];\n")
            fpoints.write("pred = [")
            fpoints.write(" ".join(str(y) for y in pred_velocities_plot0))
            fpoints.write("];\n")

        plt.plot(t_plot, velocities_plot1, label=r'True $v_y$')
        plt.plot(t_plot, pred_velocities_plot1, label=r'Pred $v_y$')
        # Add vertical lines at multiples of shooting_time
        for st in np.arange(t[0], max(t), shooting_time):
            plt.axvline(x=st, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Time [s]')
        plt.ylabel('Y Linear Velocity [m/s]')
        plt.legend()
        plt.savefig(os.path.join(directory1, file_path11)+'.png', bbox_inches='tight')
        plt.savefig(os.path.join(directory2, file_path11)+'.pdf', bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        # Saving data points
        file_points = directory3 + '/vel_y_shooting.txt'
        with open(file_points, "w") as fpoints:
            fpoints.write("x = [")
            fpoints.write(" ".join(str(x) for x in t_plot))
            fpoints.write("];\n")
            fpoints.write("true = [")
            fpoints.write(" ".join(str(y) for y in velocities_plot1))
            fpoints.write("];\n")
            fpoints.write("pred = [")
            fpoints.write(" ".join(str(y) for y in pred_velocities_plot1))
            fpoints.write("];\n")

        plt.plot(t_plot, velocities_plot2, label=r'True $v_z$')
        plt.plot(t_plot, pred_velocities_plot2, label=r'Pred $v_z$')
        # Add vertical lines at multiples of shooting_time
        for st in np.arange(t[0], max(t), shooting_time):
            plt.axvline(x=st, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Time [s]')
        plt.ylabel('Z Linear Velocity [m/s]')
        plt.legend()
        plt.savefig(os.path.join(directory1, file_path12)+'.png', bbox_inches='tight')
        plt.savefig(os.path.join(directory2, file_path12)+'.pdf', bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        # Saving data points
        file_points = directory3 + '/vel_z_shooting.txt'
        with open(file_points, "w") as fpoints:
            fpoints.write("x = [")
            fpoints.write(" ".join(str(x) for x in t_plot))
            fpoints.write("];\n")
            fpoints.write("true = [")
            fpoints.write(" ".join(str(y) for y in velocities_plot2))
            fpoints.write("];\n")
            fpoints.write("pred = [")
            fpoints.write(" ".join(str(y) for y in pred_velocities_plot2))
            fpoints.write("];\n")

        # Compute RMSE for each angular velocity component
        vel_x_rmse = compute_rmse(pred_velocities[:, 0], velocities[:, 0])
        vel_y_rmse = compute_rmse(pred_velocities[:, 1], velocities[:, 1])
        vel_z_rmse = compute_rmse(pred_velocities[:, 2], velocities[:, 2])
        mean_vel_rmse = np.mean([vel_x_rmse, vel_y_rmse, vel_z_rmse])
        file.write(f"vel x rmse: {vel_x_rmse}\n")
        file.write(f"vel y rmse: {vel_y_rmse}\n")
        file.write(f"vel z rmse: {vel_z_rmse}\n")


        # Attitude Shooting ------------------------------------------------------------------
        plt.plot(t_plot, phi_angles_plot, label=r'True $\phi$')
        plt.plot(t_plot, pred_phi_angles_plot, label=r'Pred $\phi$')
        # Add vertical lines at multiples of shooting_time
        for st in np.arange(t[0], max(t), shooting_time):
            plt.axvline(x=st, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Time [s]')
        plt.ylabel('Phi Angle [rad]')
        plt.legend()
        plt.savefig(os.path.join(directory1, file_path1)+'.png', bbox_inches='tight')
        plt.savefig(os.path.join(directory2, file_path1)+'.pdf', bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        # Saving data points
        file_points = directory3 + '/phi_shooting.txt'
        with open(file_points, "w") as fpoints:
            fpoints.write("x = [")
            fpoints.write(" ".join(str(x) for x in t_plot))
            fpoints.write("];\n")
            fpoints.write("true = [")
            fpoints.write(" ".join(str(y) for y in phi_angles_plot))
            fpoints.write("];\n")
            fpoints.write("pred = [")
            fpoints.write(" ".join(str(y) for y in pred_phi_angles_plot))
            fpoints.write("];\n")

        plt.plot(t_plot, theta_angles_plot, label=r'True $\theta$')
        plt.plot(t_plot, pred_theta_angles_plot, label=r'Pred $\theta$')
        # Add vertical lines at multiples of shooting_time
        for st in np.arange(t[0], max(t), shooting_time):
            plt.axvline(x=st, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Time [s]')
        plt.ylabel('Theta Angle [rad]')
        plt.legend()
        plt.savefig(os.path.join(directory1, file_path2)+'.png', bbox_inches='tight')
        plt.savefig(os.path.join(directory2, file_path2)+'.pdf', bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        # Saving data points
        file_points = directory3 + '/theta_shooting.txt'
        with open(file_points, "w") as fpoints:
            fpoints.write("x = [")
            fpoints.write(" ".join(str(x) for x in t_plot))
            fpoints.write("];\n")
            fpoints.write("true = [")
            fpoints.write(" ".join(str(y) for y in theta_angles_plot))
            fpoints.write("];\n")
            fpoints.write("pred = [")
            fpoints.write(" ".join(str(y) for y in pred_theta_angles_plot))
            fpoints.write("];\n")

        plt.plot(t_plot, psi_angles_plot, label=r'True $\psi$')
        plt.plot(t_plot, pred_psi_angles_plot, label=r'Pred $\psi$')
        # Add vertical lines at multiples of shooting_time
        for st in np.arange(t[0], max(t), shooting_time):
            plt.axvline(x=st, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Time [s]')
        plt.ylabel('Psi Angle [rad]')
        plt.legend()
        plt.savefig(os.path.join(directory1, file_path3)+'.png', bbox_inches='tight')
        plt.savefig(os.path.join(directory2, file_path3)+'.pdf', bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        # Saving data points
        file_points = directory3 + '/psi_shooting.txt'
        with open(file_points, "w") as fpoints:
            fpoints.write("x = [")
            fpoints.write(" ".join(str(x) for x in t_plot))
            fpoints.write("];\n")
            fpoints.write("true = [")
            fpoints.write(" ".join(str(y) for y in psi_angles_plot))
            fpoints.write("];\n")
            fpoints.write("pred = [")
            fpoints.write(" ".join(str(y) for y in pred_psi_angles_plot))
            fpoints.write("];\n")

        # Compute RMSE for each angle
        phi_rmse = compute_rmse(pred_phi_angles, phi_angles)
        theta_rmse = compute_rmse(pred_theta_angles, theta_angles)
        psi_mse = compute_rmse(pred_psi_angles, psi_angles)
        mean_att_rmse = np.mean([phi_rmse, theta_rmse, psi_mse])
        file.write(f"phi rmse: {phi_rmse}\n")
        file.write(f"theta rmse: {theta_rmse}\n")
        file.write(f"psi rmse: {psi_mse}\n")

        # Angular velocity shooting ------------------------------------------------------------------
        plt.plot(t_plot, ang_v_plot0, label=r'True $\omega_x$')
        plt.plot(t_plot, pred_ang_v_plot0, label=r'Pred $\omega_x$')
        # Add vertical lines at multiples of shooting_time
        for st in np.arange(t[0], max(t), shooting_time):
            plt.axvline(x=st, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Time [s]')
        plt.ylabel('X Angular Velocity [rad/s]')
        plt.legend()
        plt.savefig(os.path.join(directory1, file_path4)+'.png', bbox_inches='tight')
        plt.savefig(os.path.join(directory2, file_path4)+'.pdf', bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        # Saving data points
        file_points = directory3 + '/angv_x_shooting.txt'
        with open(file_points, "w") as fpoints:
            fpoints.write("x = [")
            fpoints.write(" ".join(str(x) for x in t_plot))
            fpoints.write("];\n")
            fpoints.write("true = [")
            fpoints.write(" ".join(str(y) for y in ang_v_plot0))
            fpoints.write("];\n")
            fpoints.write("pred = [")
            fpoints.write(" ".join(str(y) for y in pred_ang_v_plot0))
            fpoints.write("];\n")

        plt.plot(t_plot, ang_v_plot1, label=r'True $\omega_y$')
        plt.plot(t_plot, pred_ang_v_plot1, label=r'Pred $\omega_y$')
        # Add vertical lines at multiples of shooting_time
        for st in np.arange(t[0], max(t), shooting_time):
            plt.axvline(x=st, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Time [s]')
        plt.ylabel('Y Angular Velocity [rad/s]')
        plt.legend()
        plt.savefig(os.path.join(directory1, file_path5)+'.png', bbox_inches='tight')
        plt.savefig(os.path.join(directory2, file_path5)+'.pdf', bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        # Saving data points
        file_points = directory3 + '/angv_y_shooting.txt'
        with open(file_points, "w") as fpoints:
            fpoints.write("x = [")
            fpoints.write(" ".join(str(x) for x in t_plot))
            fpoints.write("];\n")
            fpoints.write("true = [")
            fpoints.write(" ".join(str(y) for y in ang_v_plot1))
            fpoints.write("];\n")
            fpoints.write("pred = [")
            fpoints.write(" ".join(str(y) for y in pred_ang_v_plot1))
            fpoints.write("];\n")

        plt.plot(t_plot, ang_v_plot2, label=r'True $\omega_z$')
        plt.plot(t_plot, pred_ang_v_plot2, label=r'Pred $\omega_z$')
        # Add vertical lines at multiples of shooting_time
        for st in np.arange(t[0], max(t), shooting_time):
            plt.axvline(x=st, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Time [s]')
        plt.ylabel('Z Angular Velocity [rad/s]')
        plt.legend()
        plt.savefig(os.path.join(directory1, file_path6)+'.png', bbox_inches='tight')
        plt.savefig(os.path.join(directory2, file_path6)+'.pdf', bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        # Saving data points
        file_points = directory3 + '/angv_z_shooting.txt'
        with open(file_points, "w") as fpoints:
            fpoints.write("x = [")
            fpoints.write(" ".join(str(x) for x in t_plot))
            fpoints.write("];\n")
            fpoints.write("true = [")
            fpoints.write(" ".join(str(y) for y in ang_v_plot2))
            fpoints.write("];\n")
            fpoints.write("pred = [")
            fpoints.write(" ".join(str(y) for y in pred_ang_v_plot2))
            fpoints.write("];\n")

        # Compute RMSE for each angular velocity component
        ang_v_x_rmse = compute_rmse(pred_ang_v[:, 0], ang_v[:, 0])
        ang_v_y_rmse = compute_rmse(pred_ang_v[:, 1], ang_v[:, 1])
        ang_v_z_rmse = compute_rmse(pred_ang_v[:, 2], ang_v[:, 2])
        mean_ang_v_rmse = np.mean([ang_v_x_rmse, ang_v_y_rmse, ang_v_z_rmse])
        file.write(f"ang_v x rmse: {ang_v_x_rmse}\n")
        file.write(f"ang_v y rmse: {ang_v_y_rmse}\n")
        file.write(f"ang_v z rmse: {ang_v_z_rmse}\n")
        
        # Mean RMSEs ------------------------------------------------------------------
        file.write(f"mean positions rmse: {mean_pos_rmse}\n")
        file.write(f"mean velocities rmse: {mean_vel_rmse}\n")
        file.write(f"mean attitudes rmse: {mean_att_rmse}\n")
        file.write(f"mean ang_v rmse: {mean_ang_v_rmse}\n")
        if i == 0: #Training dataset performances
            mean_training_pos_rmse = mean_pos_rmse
            mean_training_vel_rmse = mean_vel_rmse
            mean_training_att_rmse = mean_att_rmse
            mean_training_ang_v_rmse = mean_ang_v_rmse
        else: #Test datasets performances
            mean_test_pos_rmse_vec.append(mean_pos_rmse)
            mean_test_vel_rmse_vec.append(mean_vel_rmse)
            mean_test_att_rmse_vec.append(mean_att_rmse)
            mean_test_ang_v_rmse_vec.append(mean_ang_v_rmse)

        # Plot quaterniosn norm across time ------------------------------------------------------------------
        plt.plot(t_plot, pred_quaternions_norms_plot)
        plt.xlabel('Time [s]')
        plt.ylabel('Quaternions Norm')
        plt.savefig(os.path.join(directory1, file_path13)+'.png', bbox_inches='tight')
        plt.savefig(os.path.join(directory2, file_path13)+'.pdf', bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        # Saving data points
        file_points = directory3 + '/quat_norm_shooting.txt'
        with open(file_points, "w") as fpoints:
            fpoints.write("x = [")
            fpoints.write(" ".join(str(x) for x in t_plot))
            fpoints.write("];\n")
            fpoints.write("y = [")
            fpoints.write(" ".join(str(y) for y in pred_quaternions_norms_plot))
            fpoints.write("];\n")

        file.write("\n")
    
    file.write(f"\nTraining dataset mean performances:\n")
    file.write(f"mean positions rmse: {mean_training_pos_rmse}\n")
    file.write(f"mean velocities rmse: {mean_training_vel_rmse}\n")
    file.write(f"mean attitudes rmse: {mean_training_att_rmse}\n")
    file.write(f"mean ang_v rmse: {mean_training_ang_v_rmse}\n\n")

    file.write(f"Test datasets mean performances:\n")
    file.write(f"mean positions rmse: {np.mean(mean_test_pos_rmse_vec)}\n")
    file.write(f"mean velocities rmse: {np.mean(mean_test_vel_rmse_vec)}\n")
    file.write(f"mean attitudes rmse: {np.mean(mean_test_att_rmse_vec)}\n")
    file.write(f"mean ang_v rmse: {np.mean(mean_test_ang_v_rmse_vec)}\n")

    
#--------------------------------------------------------------MAIN--------------------------------------------------------------

if __name__ == "__main__":

    # Output string for the removed interval around each actuaction change
    if t_u_flag:
        str_t_u_flag = '_tu'+str(t_u_flag)+'b'+str(t_u_before)+'a'+str(t_u_after)
    else:
        str_t_u_flag = '_tu'+str(t_u_flag)

    # Read sample data
    t, positions, velocities, ang_v, quaternions, actuation = read_data(file_path)

    # Setting time to start on zero
    t = t - t[0]

    # Selecting specific portions of the dataset with the timestamps t1 and t2
    if select_data == 1:
        t, positions, velocities, ang_v, quaternions, actuation = select_data_range(t, positions, velocities, ang_v, quaternions, actuation, t_select_data_initial, t_select_data_final)

    # Only consider samples, to then be predicted, according to the pred_samples_factor
    if pred_samples_factor > 0 and isinstance(pred_samples_factor, int):
        t = t[::pred_samples_factor]
        positions = positions[::pred_samples_factor]
        velocities = velocities[::pred_samples_factor]
        ang_v = ang_v[::pred_samples_factor]
        quaternions = quaternions[::pred_samples_factor]
        actuation = actuation[::pred_samples_factor]
    else:
        print("\nerror pred_samples_factor it's not an integer.")
        sys.exit()

    # Remove true samples before and after actuation changes to avoid errors in the predictions and learning
    if t_u_flag:
        t, positions, velocities, ang_v, quaternions, actuation = filter_tensors_on_actuation_change(t, positions, velocities, ang_v, quaternions, actuation, t_u_before, t_u_after)
    t_np = t.detach().cpu().numpy()


    # Get the size of dataset
    data_size = t.size()[0]
    time_interval = (t[-1]-t[0]).detach().cpu().numpy()
    print("\nSize of the dataset: ",data_size,"samples")
    print("Total time interval of the dataset: ",time_interval,"seconds")

    # Printing total number of chunks
    print("time per each shoot:", shooting_time,"seconds")
    print("number of chunks (assuming no gaps in the dataset):", math.ceil(time_interval / shooting_time),"chunks")

    # Load computed physical parameters
    ms_sim_L = np.linalg.cholesky(ms_sim_J1)
    ms_sim_L = np.array([ms_sim_L[0][0],ms_sim_L[1][0],ms_sim_L[1][1],ms_sim_L[2][0],ms_sim_L[2][1],ms_sim_L[2][2]])
    initial_params_list = [ms_sim_L, ms_sim_c, ms_sim_A1M, ms_sim_A1F]
    initial_params = np.concatenate([param.flatten() for param in initial_params_list])

    # Initialize the NN model
    if nn_model_id == 1:
        nn_model = DynamicsMLP1(input_dim=24, hidden_dim=hidden_dim, output_dim=6)
    elif nn_model_id == 2:
        nn_model = DynamicsMLP2(input_dim=24, hidden_dim=hidden_dim, output_dim=6)
    elif nn_model_id == 3:
        nn_model = DynamicsMLP3(input_dim=24, hidden_dim=hidden_dim, output_dim=6)
    elif nn_model_id == 4:
        nn_model = DynamicsMLP4(input_dim=24, hidden_dim=hidden_dim, output_dim=6)
    elif nn_model_id == 5:
        nn_model = DynamicsMLP5(input_dim=24, hidden_dim=hidden_dim, output_dim=6)
    elif nn_model_id == 6:
        nn_model = DynamicsMLP6(input_dim=24, hidden_dim=hidden_dim, output_dim=6)
    else:
        print("\nerror nn_model_id is not valid.")
        sys.exit()

    # Move the model to the appropriate device (GPU or CPU) and tensor, to speed up the training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn_model.to(device)
    # Move all tensors to device
    t = t.to(device)
    positions = positions.to(device)
    velocities = velocities.to(device)
    ang_v = ang_v.to(device)
    quaternions = quaternions.to(device)
    actuation = actuation.to(device)

    physical_params = torch.nn.Parameter(torch.tensor(initial_params, dtype=torch.float32, device=device))

    # Initiliaze the optimizer used
    optimizer = torch.optim.AdamW(list(nn_model.parameters()) + [physical_params], lr=lr, weight_decay=weight_decay)

    # Scheduler for the learning rate
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

    # Creating the file to write the report of the training
    directory = 'outputs/'+ 'solver_'+ str(type(optimizer).__name__)+'_lr_'+str(lr)+'_epochs_'+str(num_epochs)+'_st_'+str(shooting_time)+'_taf_'+str(pred_samples_factor)+'_sch='+str(flag_scheduler) +str_t_u_flag
    os.makedirs(directory, exist_ok=True)
    file_name = '/solver_'+ str(type(optimizer).__name__)+'_lr_'+str(lr)+'_epochs_'+str(num_epochs)+'_st_'+str(shooting_time)+'_taf_'+str(pred_samples_factor)+'_sch='+str(flag_scheduler)+str_t_u_flag
    file_name = directory+file_name+".txt"
    file = open(file_name, "w")

    print("solver: ", type(optimizer).__name__)
    print(f"lr: {lr}\n")
    file.write("solver: " + str(type(optimizer).__name__) + "\n")
    file.write("lr: " + str(lr) + "\n")
    file.write("num_epochs: " + str(num_epochs) + "\n")
    file.write("weight_decay: " + str(weight_decay) + "\n")
    file.write("file_path: " + str(file_path) + "\n")
    file.write("lambda_nn_reg: " + str(lambda_nn_reg) + "\n")
    file.write("plot_hparam_pos: " + str(hparam_pos) + "\n")
    file.write("plot_hparam_vel: " + str(hparam_vel) + "\n")
    file.write("plot_hparam_quat: " + str(hparam_quat) + "\n")
    file.write("plot_hparam_ang_v: " + str(hparam_ang_v) + "\n")
    file.write("nn_model_id: " + str(nn_model_id) + "\n")
    file.write("hidden_dim: " + str(hidden_dim) + "\n")
    file.write("validation_file_path: " + str(validation_file_path) + "\n")
    file.write("warmup_epochs: " + str(warmup_epochs) + "\n")
    file.write("patience: " + str(patience) + "\n")
    file.write("early_stop: " + str(early_stop) + "\n")
    file.write("best_valloss_model: " + str(best_valloss_model) + "\n")
    file.write("flag_validation: " + str(flag_validation) + "\n")
    file.write("percentage_consider_chunk: " + str(percentage_consider_chunk) + "\n")
    file.write("percentage_validation: " + str(percentage_validation) + "\n")
    file.write("validation_chunks: " + str(validation_chunks) + "\n")
    file.write("just_last_sample: " + str(just_last_sample) + "\n")
    file.write("rtol_aux: " + str(rtol_aux) + "\n")
    file.write("atol_aux: " + str(atol_aux) + "\n")
    file.write("shooting_time: " + str(shooting_time) + "\n")
    file.write("pred_samples_factor: " + str(pred_samples_factor) + "\n")
    file.write("flag_scheduler: " + str(flag_scheduler) + "\n")
    file.write("t_u_flag: " + str(t_u_flag) + "\n")
    file.write("t_u_before: " + str(t_u_before) + "\n")
    file.write("t_u_after: " + str(t_u_after) + "\n\n")

    total_loss_vec = []
    mean_total_loss_vec = []

    #auxiliar time vectors
    vec_time_odeint = []
    vec_time_shooting = []
    vec_time_learning = []

    # Early stopping, open validation set and read data
    if flag_validation == 0: #Load new dataset for validation
        t_validation, positions_validation, velocities_validation, ang_v_validation, quaternions_validation, actuation_validation = read_data(validation_file_path)
        t_validation = t_validation - t_validation[0]
        t_validation_np = t_validation.detach().cpu().numpy()
    elif flag_validation == 1 or flag_validation == 2: #Use part of the training set as validation set
        data = split_val_train_chunks(t, positions, velocities, ang_v, quaternions, actuation, t_separation, shooting_time, percentage_validation, percentage_consider_chunk, flag_validation, validation_chunks)
        print(f"Segments found: {len(data['segments_found'])}")
        file.write("Segments found: "  + str(len(data['segments_found'])) + "\n")
        print("Segment indices (start, end):")
        file.write("Segment indices (start, end):\n")
        for seg in data['segments_found']:
            print(f"  {seg}")
            file.write(str({seg})+ "\n")
        print(f"Chosen validation segment indices: {[i + 1 for i in sorted(data['chosen_validation_indices'])]}")
        file.write("Chosen validation segment indices: "  + str([i + 1 for i in sorted(data['chosen_validation_indices'])]) + "\n")
        print(f"Train set shapes: {len(data['t_train'])}")
        file.write("Train set shapes: "  + str(len(data['t_train'])) + "\n")
        print(f"Validation set shapes: {len(data['t_val'])}\n")
        file.write("Validation set shapes: "  + str(len(data['t_val'])) + "\n\n")
        t = data['t_train']
        positions = data['positions_train']
        velocities = data['velocities_train']
        ang_v = data['ang_v_train']
        quaternions = data['quaternions_train']
        actuation = data['actuation_train']
        t_np = t.detach().cpu().numpy()
        t_validation = data['t_val']
        positions_validation = data['positions_val']
        velocities_validation = data['velocities_val']
        ang_v_validation = data['ang_v_val']
        quaternions_validation = data['quaternions_val']
        actuation_validation = data['actuation_val']
        t_validation_np = t_validation.detach().cpu().numpy()

    # Defining sizes in samples of training and validation sets
    data_size_training = t.size()[0]
    data_size_validation = t_validation.size()[0]

    best_validation_loss = 0
    best_model_state_dict = None
    best_epoch = 0
    epochs_no_improve = 0
    validation_loss_vec = []
    mean_validation_loss_vec = []

    # Move the model to the appropriate device (GPU or CPU) and tensor, to speed up the training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_validation = t_validation.to(device)
    positions_validation = positions_validation.to(device)
    velocities_validation = velocities_validation.to(device)
    ang_v_validation = ang_v_validation.to(device)
    quaternions_validation = quaternions_validation.to(device)
    actuation_validation = actuation_validation.to(device)

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Multiple shooting function, forward propagation
        start_shooting_time = time.time()    
        total_loss,_,_,_,_,odeint_time = shooting(nn_model, t, t_np, positions, velocities, ang_v, quaternions, actuation, just_last_sample, rtol_aux, atol_aux, shooting_time, hparam_pos, hparam_vel, hparam_quat, hparam_ang_v, t_separation, physical_params, lambda_nn_reg)
        end_shooting_time = time.time()
        total_loss_vec.append(total_loss.item())
        vec_time_odeint.append(odeint_time)
        vec_time_shooting.append(end_shooting_time - start_shooting_time)
        print("shooting time: ",end_shooting_time - start_shooting_time)

        # Validation Loss, Best model according to it and Early stopping
        validation_loss,_,_,_,_,_ = shooting(nn_model, t_validation, t_validation_np, positions_validation, velocities_validation, ang_v_validation, quaternions_validation, actuation_validation, just_last_sample, rtol_aux, atol_aux, shooting_time, hparam_pos, hparam_vel, hparam_quat, hparam_ang_v, t_separation, physical_params, lambda_nn_reg)
        validation_loss_vec.append(validation_loss.item())
        if epoch == 0:
            best_validation_loss = validation_loss.item()
        if epoch >= warmup_epochs:
            if validation_loss.item() < best_validation_loss:
                best_validation_loss = validation_loss.item()
                best_model_state_dict = copy.deepcopy(nn_model.state_dict())
                best_physical_params = physical_params.clone().detach()
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
        if epochs_no_improve >= patience and early_stop==1:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            file.write("\nEarly stopping triggered at epoch "  + str(epoch) + "\n")
            break

        start_time1 = time.time()
        total_loss.backward()
        optimizer.step()
        if flag_scheduler == 1:
            scheduler.step(total_loss)
        end_time1 = time.time()
        vec_time_learning.append(end_time1 - start_time1)

        # Print the current learning rate after optimizer step
        for param_group in optimizer.param_groups:
            print(f"Current learning rate: {param_group['lr']}")
            file.write("Learning Rate: " + str(param_group['lr']) + "\n")

        # Compute mean training and validation losses per sample
        mean_total_loss = total_loss / data_size_training
        mean_validation_loss = validation_loss / data_size_validation
        mean_total_loss_vec.append(mean_total_loss.item())
        mean_validation_loss_vec.append(mean_validation_loss.item())

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {total_loss.item()}, Validation Loss: {validation_loss.item()}")
        print(f"Mean Training Loss: {mean_total_loss.item()}, Mean Validation Loss: {mean_validation_loss.item()}\n")
        file.write("Epoch " + str(epoch+1) + "/" + str(num_epochs) + ", Training Loss: " + str(total_loss.item()) + ", Validation Loss: " + str(validation_loss.item()) + "\n")
        file.write("Mean Training Loss: " + str(mean_total_loss.item()) + ", Mean Validation Loss: " + str(mean_validation_loss.item()) + "\n\n")

    # Save the final physical parameters
    final_n_ms_sim_L, final_c, final_A1M, final_A1F = recover_params(physical_params)
    final_matrix_n_ms_sim_L = final_n_ms_sim_L.detach().numpy()
    final_matrix_n_ms_sim_J1 = final_matrix_n_ms_sim_L @ final_matrix_n_ms_sim_L.T
    final_J1 = torch.tensor(final_matrix_n_ms_sim_J1)
    folder = 'solver_'+ str(type(optimizer).__name__)+'_lr_'+str(lr)+'_epochs_'+str(num_epochs)+'_st_'+str(shooting_time)+'_taf_'+str(pred_samples_factor)+'_sch='+str(flag_scheduler)+str_t_u_flag
    directory = 'outputs/'+folder
    os.makedirs(directory, exist_ok=True)
    output_filename1 = 'J1_epoch.npy'
    output_filename2 = 'c_epoch.npy'
    output_filename3 = 'A1M_epoch.npy'
    output_filename4 = 'A1F_epoch.npy'
    file_path1 = os.path.join(directory, output_filename1)
    file_path2 = os.path.join(directory, output_filename2)
    file_path3 = os.path.join(directory, output_filename3)
    file_path4 = os.path.join(directory, output_filename4)
    np.save(file_path1, final_J1.detach().cpu().numpy())
    np.save(file_path2, final_c.detach().cpu().numpy())
    np.save(file_path3, final_A1M.detach().cpu().numpy())
    np.save(file_path4, final_A1F.detach().cpu().numpy()) 

    # Save the best physical parameters according to validation loss
    if best_model_state_dict is not None and best_valloss_model==1:
        best_n_ms_sim_L, best_c, best_A1M, best_A1F = recover_params(best_physical_params)
        best_matrix_n_ms_sim_L = best_n_ms_sim_L.detach().numpy()
        best_matrix_n_ms_sim_J1 = best_matrix_n_ms_sim_L @ best_matrix_n_ms_sim_L.T
        best_J1 = torch.tensor(best_matrix_n_ms_sim_J1)
        output_filename1 = 'J1_best_epoch.npy'
        output_filename2 = 'c_best_epoch.npy'
        output_filename3 = 'A1M_best_epoch.npy'
        output_filename4 = 'A1F_best_epoch.npy'
        file_path1 = os.path.join(directory, output_filename1)
        file_path2 = os.path.join(directory, output_filename2)
        file_path3 = os.path.join(directory, output_filename3)
        file_path4 = os.path.join(directory, output_filename4)
        np.save(file_path1, best_J1.detach().cpu().numpy())
        np.save(file_path2, best_c.detach().cpu().numpy())
        np.save(file_path3, best_A1M.detach().cpu().numpy())
        np.save(file_path4, best_A1F.detach().cpu().numpy())
        
    # Recompute loss with final updated params
    final_loss,_,_,_,_,_ = shooting(nn_model, t, t_np, positions, velocities, ang_v, quaternions, actuation, just_last_sample, rtol_aux, atol_aux, shooting_time, hparam_pos, hparam_vel, hparam_quat, hparam_ang_v, t_separation, physical_params, lambda_nn_reg)
    print("Final training loss with updated params: ", final_loss.item())
    file.write("Final training loss with updated params: " + str(final_loss.item()) + "\n")

    # Final Validation Loss computation and possible best model update
    validation_loss,_,_,_,_,_ = shooting(nn_model, t_validation, t_validation_np, positions_validation, velocities_validation, ang_v_validation, quaternions_validation, actuation_validation, just_last_sample, rtol_aux, atol_aux, shooting_time, hparam_pos, hparam_vel, hparam_quat, hparam_ang_v, t_separation, physical_params, lambda_nn_reg)
    print("Final validation loss with updated params: ", validation_loss.item())
    file.write("Final validation loss with updated params: " + str(validation_loss.item()) + "\n")
    # Best model update
    if validation_loss.item() < best_validation_loss:
        best_validation_loss = validation_loss.item()
        best_model_state_dict = copy.deepcopy(nn_model.state_dict())
        best_epoch = epoch

    # Epoch of the best model, with lower validation loss
    if best_model_state_dict is not None and best_valloss_model==1:
        print(f"Best model, with lower validation loss, found at epoch {best_epoch+1}\n")
        file.write("Best model, with lower validation loss, found at epoch "  + str(best_epoch+1) + "\n")

    # Create a directory to save plot images
    directory1 = 'outputs/'+ 'solver_'+ str(type(optimizer).__name__)+'_lr_'+str(lr)+'_epochs_'+str(num_epochs)+'_st_'+str(shooting_time)+'_taf_'+str(pred_samples_factor)+'_sch='+str(flag_scheduler) +str_t_u_flag
    directory2 = 'outputs/'+ 'solver_'+ str(type(optimizer).__name__)+'_lr_'+str(lr)+'_epochs_'+str(num_epochs)+'_st_'+str(shooting_time)+'_taf_'+str(pred_samples_factor)+'_sch='+str(flag_scheduler) +str_t_u_flag + '/loss_pdf_data'
    os.makedirs(directory1, exist_ok=True)
    os.makedirs(directory2, exist_ok=True)
    file_path1 = 'att_train_loss_plot_solver_'+ str(type(optimizer).__name__)+'_lr_'+str(lr)+'_epochs_'+str(num_epochs)+'_st_'+str(shooting_time)+'_taf_'+str(pred_samples_factor)+'_sch='+str(flag_scheduler) +str_t_u_flag
    file_path2 = 'att_train_val_loss_plot_solver_'+ str(type(optimizer).__name__)+'_lr_'+str(lr)+'_epochs_'+str(num_epochs)+'_st_'+str(shooting_time)+'_taf_'+str(pred_samples_factor)+'_sch='+str(flag_scheduler) +str_t_u_flag
    file_path3 = 'att_mean_train_loss_plot_solver_'+ str(type(optimizer).__name__)+'_lr_'+str(lr)+'_epochs_'+str(num_epochs)+'_st_'+str(shooting_time)+'_taf_'+str(pred_samples_factor)+'_sch='+str(flag_scheduler) +str_t_u_flag
    file_path4 = 'att_mean_train_val_loss_plot_solver_'+ str(type(optimizer).__name__)+'_lr_'+str(lr)+'_epochs_'+str(num_epochs)+'_st_'+str(shooting_time)+'_taf_'+str(pred_samples_factor)+'_sch='+str(flag_scheduler) +str_t_u_flag
    epochs_ran = len(total_loss_vec)
    # Only Training Loss Plot
    plt.plot(range(epochs_ran), total_loss_vec)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)  # Optional: helps with readability
    plt.savefig(os.path.join(directory1, file_path1)+'.png', bbox_inches='tight')
    plt.savefig(os.path.join(directory2, file_path1)+'.pdf', bbox_inches='tight')
    plt.close()
    # Saving data points
    file_points = directory2 + '/loss.txt'
    with open(file_points, "w") as fpoints:
        fpoints.write("x = [")
        fpoints.write(" ".join(str(x) for x in range(epochs_ran)))
        fpoints.write("];\n")
        fpoints.write("y = [")
        fpoints.write(" ".join(str(y) for y in total_loss_vec))
        fpoints.write("];\n")
    # Training and Validation Losses Plot
    plt.plot(range(epochs_ran), total_loss_vec, label='Training Loss')
    plt.plot(range(epochs_ran), validation_loss_vec, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)  # Optional: helps with readability
    plt.savefig(os.path.join(directory1, file_path2)+'.png', bbox_inches='tight')
    plt.savefig(os.path.join(directory2, file_path2)+'.pdf', bbox_inches='tight')
    plt.close()
    # Saving data points
    file_points = directory2 + '/train_val_losses.txt'
    with open(file_points, "w") as fpoints:
        fpoints.write("x = [")
        fpoints.write(" ".join(str(x) for x in range(epochs_ran)))
        fpoints.write("];\n")
        fpoints.write("train = [")
        fpoints.write(" ".join(str(y) for y in total_loss_vec))
        fpoints.write("];\n")
        fpoints.write("val = [")
        fpoints.write(" ".join(str(y) for y in validation_loss_vec))
        fpoints.write("];\n")
    # Only Mean Training Loss Plot
    plt.plot(range(epochs_ran), mean_total_loss_vec)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Loss')
    plt.grid(True)  # Optional: helps with readability
    plt.savefig(os.path.join(directory1, file_path3)+'.png', bbox_inches='tight')
    plt.savefig(os.path.join(directory2, file_path3)+'.pdf', bbox_inches='tight')
    plt.close()
    # Saving data points
    file_points = directory2 + '/mean_loss.txt'
    with open(file_points, "w") as fpoints:
        fpoints.write("x = [")
        fpoints.write(" ".join(str(x) for x in range(epochs_ran)))
        fpoints.write("];\n")
        fpoints.write("y = [")
        fpoints.write(" ".join(str(y) for y in mean_total_loss_vec))
        fpoints.write("];\n")
    # Mean Training and Validation Losses Plot
    plt.plot(range(epochs_ran), mean_total_loss_vec, label='Mean Training Loss')
    plt.plot(range(epochs_ran), mean_validation_loss_vec, label='Mean Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Loss')
    plt.legend()
    plt.grid(True)  # Optional: helps with readability
    plt.savefig(os.path.join(directory1, file_path4)+'.png', bbox_inches='tight')
    plt.savefig(os.path.join(directory2, file_path4)+'.pdf', bbox_inches='tight')
    plt.close()
    # Saving data points
    file_points = directory2 + '/mean_train_val_losses.txt'
    with open(file_points, "w") as fpoints:
        fpoints.write("x = [")
        fpoints.write(" ".join(str(x) for x in range(epochs_ran)))
        fpoints.write("];\n")
        fpoints.write("train = [")
        fpoints.write(" ".join(str(y) for y in mean_total_loss_vec))
        fpoints.write("];\n")
        fpoints.write("val = [")
        fpoints.write(" ".join(str(y) for y in mean_validation_loss_vec))
        fpoints.write("];\n")

    # Save the model
    output_model_filename = 'nn_model_'+ str(type(optimizer).__name__)+'_lr='+str(lr)+'_epochs_'+str(num_epochs)+'_st='+str(shooting_time)+'_taf_'+str(pred_samples_factor)+'_sch='+str(flag_scheduler) +str_t_u_flag +'.pth'
    os.makedirs(directory, exist_ok=True)
    output_model_file_path = os.path.join(directory, output_model_filename)
    torch.save(nn_model.state_dict(), output_model_file_path)

    # Save also the best model, the one with lower validation loss
    if best_model_state_dict is not None and best_valloss_model==1:
        # Initialize the best NN model
        if nn_model_id == 1:
            best_nn_model = DynamicsMLP1(input_dim=24, hidden_dim=hidden_dim, output_dim=6)
        elif nn_model_id == 2:
            best_nn_model = DynamicsMLP2(input_dim=24, hidden_dim=hidden_dim, output_dim=6)
        elif nn_model_id == 3:
            best_nn_model = DynamicsMLP3(input_dim=24, hidden_dim=hidden_dim, output_dim=6)
        elif nn_model_id == 4:
            best_nn_model = DynamicsMLP4(input_dim=24, hidden_dim=hidden_dim, output_dim=6)
        elif nn_model_id == 5:
            best_nn_model = DynamicsMLP5(input_dim=24, hidden_dim=hidden_dim, output_dim=6)
        elif nn_model_id == 6:
            best_nn_model = DynamicsMLP6(input_dim=24, hidden_dim=hidden_dim, output_dim=6)
        # Load the best model
        best_nn_model.load_state_dict(best_model_state_dict)
        # Save the best model model also
        output_model_filename_best = 'best_nn_model_'+ str(type(optimizer).__name__)+'_lr='+str(lr)+'_epochs_'+str(num_epochs)+'_st='+str(shooting_time)+'_taf_'+str(pred_samples_factor)+'_sch='+str(flag_scheduler) +str_t_u_flag +'.pth'
        os.makedirs(directory, exist_ok=True)
        output_model_file_path_best = os.path.join(directory, output_model_filename_best)
        torch.save(best_nn_model.state_dict(), output_model_file_path_best)
    elif best_valloss_model==1:
        print("Warning: Best model state dict was never updated!")
        file.write("Warning: Best model state dict was never updated!")

    # Close the report .txt file
    file.close()



    # ------------------------------------------------------- Plots with the final solution -------------------------------------------------------

    # Creating the results file and directory for the plots
    directory = 'outputs/'+ 'solver_'+ str(type(optimizer).__name__)+'_lr_'+str(lr)+'_epochs_'+str(num_epochs)+'_st_'+str(shooting_time)+'_taf_'+str(pred_samples_factor)+'_sch='+str(flag_scheduler) +str_t_u_flag
    os.makedirs(directory, exist_ok=True)
    file_name = directory + '/results.txt'
    file = open(file_name, "w")
    directory1 = directory + '/shooting_plots_png'
    directory2 = directory + '/shooting_plots_pdf'
    directory3 = directory + '/shooting_plots_data'
    os.makedirs(directory1, exist_ok=True)
    os.makedirs(directory2, exist_ok=True)
    os.makedirs(directory3, exist_ok=True)
    file.write("test sets: " + str(datasets_vec[1:]) + "\n\n")

    hparam_pos = plot_hparam_pos
    hparam_vel = plot_hparam_vel
    hparam_quat = plot_hparam_quat
    hparam_ang_v = plot_hparam_ang_v

    # Plots the shooting results and compute RMSE results
    plot_shooting_results_rmse(datasets_vec, plot_select_data_vec, plot_t_select_data_initial_vec, plot_t_select_data_final_vec, plot_t_u_flag_vec, plot_t_u_before_vec, plot_t_u_after_vec, nn_model, hparam_pos, hparam_vel, hparam_quat, hparam_ang_v, pred_samples_factor, shooting_time, just_last_sample, rtol_aux, atol_aux, file, directory1, directory2, directory3, num_epochs, lr, optimizer, flag_scheduler, str_t_u_flag, folder_gimble, physical_params, lambda_nn_reg)

    # Close the report .txt file
    file.close()

    if best_model_state_dict is not None and best_valloss_model==1:
        # Creating the results file and directory for the plots with the best model
        directory = 'outputs/'+ 'solver_'+ str(type(optimizer).__name__)+'_lr_'+str(lr)+'_epochs_'+str(num_epochs)+'_st_'+str(shooting_time)+'_taf_'+str(pred_samples_factor)+'_sch='+str(flag_scheduler) +str_t_u_flag
        os.makedirs(directory, exist_ok=True)
        file_name = directory + '/results_best_model.txt'
        file = open(file_name, "w")
        directory1 = directory + '/shooting_plots_png_best_model'
        directory2 = directory + '/shooting_plots_pdf_best_model'
        directory3 = directory + '/shooting_plots_data_best_model'
        os.makedirs(directory1, exist_ok=True)
        os.makedirs(directory2, exist_ok=True)
        os.makedirs(directory3, exist_ok=True)
        file.write("test sets: " + str(datasets_vec[1:]) + "\n\n")

        # Plots the shooting results and compute RMSE results
        plot_shooting_results_rmse(datasets_vec, plot_select_data_vec, plot_t_select_data_initial_vec, plot_t_select_data_final_vec, plot_t_u_flag_vec, plot_t_u_before_vec, plot_t_u_after_vec, best_nn_model, hparam_pos, hparam_vel, hparam_quat, hparam_ang_v, pred_samples_factor, shooting_time, just_last_sample, rtol_aux, atol_aux, file, directory1, directory2, directory3, num_epochs, lr, optimizer, flag_scheduler, str_t_u_flag, folder_gimble, physical_params, lambda_nn_reg)

        # Close the report .txt file
        file.close()