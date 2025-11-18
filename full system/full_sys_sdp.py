import cvxpy as cp
import numpy as np
import numpy.linalg as la
import sys
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchdiffeq import odeint
import sys
import torch.autograd.functional as F_autograd
import math
import time

DIR = os.getcwd()


# Cobot parameters
g = np.array([[0], [0], [-9.81]], dtype=float)  # gravity acceleration
m = 1  # mass of Cobot
m_batt = 0  # mass of the battery
M = m + m_batt  # total mass
r = 0.25  # radius of Cobot
h = 0.1  # height of Cobot


class Estimator:
    def __init__(self, file_path, t_u_before=0.5, t_u_after=1.0):
        self.timestamp = []
        self.p = []
        self.v = []
        self.w = []
        self.R = []
        self.u = []

        self.read_file(file_path, t_u_before=t_u_before, t_u_after=t_u_after)
        self.peprocess_data()

        # placeholders for results
        self.theta = None
        self.A1F = None
        self.A1M = None
        self.J1 = None
        self.c = None
    
    def get_S(self, v):
        """
                |   0    -v3     v2 |
        S(v) =  |  v3      0    -v1 |
                | -v2     v1      0 |
        """
        v = v.flatten()
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    
    def get_D(self, x):
        """D(x) as in the equations, x \in R^3 -> 3x6 matrix"""
        x = x.flatten()
        return np.array(
            [
                [x[0], x[1], x[2], 0.0, 0.0, 0.0],
                [0.0, x[0], 0.0, x[1], x[2], 0.0],
                [0.0, 0.0, x[0], 0.0, x[1], x[2]],
            ]
        )
    
    def phi_Jk(self, w, w_dot):
        """Phi^J_k = D(w_dot) + S(w) D(w)  (3 x 6)"""
        return self.get_D(w_dot) + self.get_S(w) @ self.get_D(w)
    

    def phi_Ak(self, u):
        """
        Returns - (I3 ⊗ u^T) as a 3x18 block (matching equation).
        The same helper is used for A_M (no R) and for A_F (multiplied by R).
        """
        u = u.flatten()
        # build row blocks: [u, 0, 0], [0, u, 0], [0, 0, u] but with negatives
        l1 = list(u) + [0.0] * 12
        l2 = [0.0] * 6 + list(u) + [0.0] * 6
        l3 = [0.0] * 12 + list(u)
        return -np.array([l1, l2, l3])

    def phi_Ck(self, Rk, w, w_dot):
        """Phi^C_k = R_k * ( S(w_dot) + S(w)^2 )  (3 x 3)"""
        Sw = self.get_S(w)
        Swdot = self.get_S(w_dot)
        Sw2 = Sw @ Sw
        return Rk @ (Swdot + Sw2)

    def get_R(self, line):
        """
        Convert a quaternion to a 3x3 rotation matrix.

        Parameters:
            quaternion: The quaternion in the format (w, x, y, z).

        Output:
            3x3 rotation matrix.
        """
        x, y, z, w = [float(x) for x in line.replace(" ", ",").split(",")]

        R = Rotation.from_quat([x, y, z, w]).as_matrix()

        return R

    def parse_line(self, line):
        """
        Convert a line of numbers to a numpy array.

        Parameters:
            line: The line of numbers.

        Output:
            An array of numbers in the line.
        """
        line = line.replace("\n", "").replace(" ", ",")
        numbers = line.strip().split(",")

        return np.array([[float(n) for n in numbers if n != ""]])

    def read_file(self, file_name, t_u_before=0.5, t_u_after=1.0):
        """
        Read the input file and store the data, discarding samples within a window
        around actuation changes.

        Parameters:
            file_name: Path to the input file
            t_u_before: Time to discard BEFORE actuation change
            t_u_after: Time to discard AFTER actuation change
        """
        print(f'Reading file "{DIR}{file_name}"')

        all_t = []
        all_p = []
        all_v = []
        all_w = []
        all_R = []
        all_u = []

        actuation_change_times = []
        prev_u = None

        with open(file_name) as fp:
            contents = fp.read()
            measurements = contents.split("\n\n")

            for m in measurements:
                m = m.split("\n")

                if len(m) == 1 and m[0] == "":
                    break

                # Parse each line
                t = float(m[0].strip())
                p = self.parse_line(m[1]).T   # position (x, y, z)
                v = self.parse_line(m[2]).T   # velocity (vx, vy, vz)
                w = self.parse_line(m[3]).T   # angular velocity
                R = self.get_R(m[4])          # quaternion -> rotation matrix
                u = self.parse_line(m[5]).T   # control inputs

                # Collect
                all_t.append(t)
                all_p.append(p)
                all_v.append(v)
                all_w.append(w)
                all_R.append(R)
                all_u.append(u)

                if prev_u is not None and not np.array_equal(u, prev_u):
                    actuation_change_times.append(t)

                prev_u = u

        # Build list of allowed indices (not in discard window)
        valid_indices = []
        for i, t in enumerate(all_t):
            discard = False
            for t_change in actuation_change_times:
                if t_change - t_u_before <= t <= t_change + t_u_after:
                    discard = True
                    break
            if not discard:
                valid_indices.append(i)

        # Apply filtering
        self.timestamp = [all_t[i] for i in valid_indices]
        self.p = [all_p[i] for i in valid_indices]
        self.v = [all_v[i] for i in valid_indices]
        self.w = [all_w[i] for i in valid_indices]
        self.R = [all_R[i] for i in valid_indices]
        self.u = [all_u[i] for i in valid_indices]

        print(f"Total samples: {len(all_t)} | Valid samples after filtering: {len(self.timestamp)}")
        return

    def peprocess_data(self, t_separation=0.051):
        """Precompute the data to be used in the cost function, with time gap check."""

        # Clear/initialize
        self.w_diff = []
        self.w_avg = []
        self.u_avg = []
        self.R_avg = []
        self.v_diff = []
        self.v_avg = []
        self.p_avg = []

        for i in range(len(self.timestamp) - 1):
            dt = self.timestamp[i + 1] - self.timestamp[i]
            if dt > t_separation:
                continue  # Skip pairs that are too far apart in time
            
            # Angular acceleration
            w_diff_i = (self.w[i + 1] - self.w[i]) / dt
            w_avg_i = (self.w[i + 1] + self.w[i]) / 2

            # Linear acceleration
            v_diff_i = (self.v[i + 1] - self.v[i]) / dt
            v_avg_i = (self.v[i + 1] + self.v[i]) / 2

            # Average position
            p_avg_i = (self.p[i + 1] + self.p[i]) / 2

            # Average input
            u_avg_i = (self.u[i + 1] + self.u[i]) / 2

            # Average rotation using SVD
            R_sum = self.R[i + 1] + self.R[i]
            U, _, Vt = np.linalg.svd(R_sum)
            R_avg_i = U @ np.diag([1, 1, np.linalg.det(U) * np.linalg.det(Vt)]) @ Vt

            # Append valid data
            self.w_diff.append(w_diff_i)
            self.w_avg.append(w_avg_i)
            self.v_diff.append(v_diff_i)
            self.v_avg.append(v_avg_i)
            self.p_avg.append(p_avg_i)
            self.u_avg.append(u_avg_i)
            self.R_avg.append(R_avg_i)
    
    def build_stacked_Phi_b(self):
        """
        Build big Phi (6K x 45) and big b (6K x 1) by stacking per-sample Phi_k and b_k.
        Theta ordering used here:
            theta = [j11, j12, j13, j22, j23, j33, c1, c2, c3, AF(18), AM(18)]^T
        AF and AM are each 3x6 flattened in row-major order.
        """
        phi_blocks = []
        b_blocks = []

        n_samples = len(self.w_avg)
        for i in range(n_samples):
            w = self.w_avg[i].flatten()
            w_dot = self.w_diff[i].flatten()
            u = self.u_avg[i].flatten()
            Rk = self.R_avg[i]

            # sub-blocks
            PhiJ = self.phi_Jk(w, w_dot)           # 3 x 6
            PhiC = self.phi_Ck(Rk, w, w_dot)       # 3 x 3
            Phi_A_base = self.phi_Ak(u)            # - (I3 \kron u^T)  3 x 18
            PhiAF = Rk @ Phi_A_base                    # 3 x 18  -> - R * (I3 \kron u^T)
            PhiAM = Phi_A_base                         # 3 x 18  -> - (I3 \kron u^T)

            top = np.hstack([np.zeros((3, 6)), PhiC, PhiAF, np.zeros((3, 18))])     # translational eqns (3 rows)
            bottom = np.hstack([PhiJ, np.zeros((3, 3)), np.zeros((3, 18)), PhiAM])  # rotational eqns (3 rows)

            Phi_k = np.vstack([top, bottom])   # 6 x 45

            v_dot = self.v_diff[i].reshape(3, 1)  # make column
            b_k = np.vstack([v_dot, np.zeros((3, 1))])  # 6 x 1

            phi_blocks.append(Phi_k)
            b_blocks.append(b_k)

        big_Phi = np.vstack(phi_blocks)   # (6K) x 45
        big_b = np.vstack(b_blocks)       # (6K) x 1

        return big_Phi, big_b
    
    def solve(self, solver=cp.SCS, verbose=False):
        """
        Build stacked matrices, run reduced QR, and solve SDP in CVXPY.
        """
        Phi, b = self.build_stacked_Phi_b()
        m, n = Phi.shape
        print(f"Built stacked Phi: shape {Phi.shape}, b: {b.shape}")

        # reduced QR: Phi = Q_r R  with Q_r (m x n), R (n x n)
        Q_r, R_mat = np.linalg.qr(Phi, mode="reduced")
        s_u = (Q_r.T @ b).flatten()   # n-vector (this is Q_r^T b)
        R_u = R_mat                   # n x n upper triangular

        # CVX variables
        th = cp.Variable(n)   # theta (45)
        t = cp.Variable()

        # Build J from first 6 elements of theta using symmetric layout:
        # order: [j11, j12, j13, j22, j23, j33]  (this matches your earlier code)
        row1 = cp.hstack([th[0], th[1], th[2]])
        row2 = cp.hstack([th[1], th[3], th[4]])
        row3 = cp.hstack([th[2], th[4], th[5]])
        Jcp = cp.vstack([row1, row2, row3])

        # c (3x1)
        c_cp = cp.reshape(th[6:9], (3, 1))

        # barM1(\theta) = [[1/2 tr(J) I - J, c],[c^T, 1]]
        M11 = 0.5 * cp.trace(Jcp) * np.eye(3) - Jcp
        barM = cp.bmat([[M11, c_cp], [c_cp.T, cp.reshape(cp.Constant(1.0), (1, 1))]])

        # Quadratic LMI: [[t, (R_u th + s_u)^T]; [R_u th + s_u, I]] >= 0
        # compute linear term R_u * th + s_u (R_u is numpy array)
        # convert s_u to constant vector
        s_const = s_u.astype(float)
        R_np = R_u.astype(float)

        r_lin = R_np @ th + s_const   # cp expression (length n)
        top_row = cp.reshape(t, (1, 1))
        top_right = cp.reshape(r_lin, (1, n))
        bottom_left = cp.reshape(r_lin, (n, 1))
        bottom_right = np.eye(n)

        bigLMI = cp.bmat([[top_row, top_right], [bottom_left, bottom_right]])

        # constraints
        constraints = [barM >> 0, bigLMI >> 0, t >= 0]
        constraints += [th[0] >= 0.01] #so that we dont get the trivial solution

        problem = cp.Problem(cp.Minimize(t), constraints)

        # solve (try SCS first, fallback to CVXOPT if available)
        try:
            problem.solve(solver=solver, verbose=verbose)
        except Exception as e:
            print("Primary solver error:", e)
            print("Retrying with SCS (default) ...")
            problem.solve(solver=cp.SCS, verbose=verbose)

        if th.value is None:
            raise RuntimeError("CVX solver failed to return a solution. Try different solver or check data rank.")

        self.theta = th.value.copy()

        # Parse results
        # J
        j11, j12, j13, j22, j23, j33 = self.theta[0:6]
        self.J1 = np.array([[j11, j12, j13], [j12, j22, j23], [j13, j23, j33]])
        # c
        self.c = self.theta[6:9].reshape((3, 1))
        # A_F and A_M
        AF_vec = self.theta[9:9 + 18]
        AM_vec = self.theta[9 + 18:9 + 36]
        self.A1F = AF_vec.reshape((3, 6))   # row-major -> each row corresponds to force rows
        self.A1M = AM_vec.reshape((3, 6))

        print("Solved. t (objective) =", t.value)
        print("Estimated J:\n", self.J1)
        print("Estimated c:\n", self.c.flatten())
        print("Estimated A_F (3x6):\n", self.A1F)
        print("Estimated A_M (3x6):\n", self.A1M)

        return self.theta

def main(argv):
    t_u_before = 0
    t_u_after = 0
    t_u_flag = 0
    folder_gimble = "full_system"
    training_dataset = "fs_sim128shots_first"
    evaluate_on_linear = 1
    # preffix_vec = [""]
    # suffix_vec = [""]
    preffix_vec = ["simnl_", "simnl_", "simnl_", "simnl_", "simnl_", "simnl_", "simnl_", "simnl_", "simnl_", "simnl_", "simnl_", "simnl_", "simnl_"]
    suffix_vec = ["_QM3", "_QM4", "_QM6", "_QM7", "_QM8", "_QM9", "_BFE1", "_BFE2", "_BFE3", "_BFE4", "_BFE5", "_AD1", "_AD2"]

    # Loss hyperparameters for each state variable
    plot_hparam_pos = 0.8127
    plot_hparam_vel = 0.1236
    plot_hparam_quat = 0.0616
    plot_hparam_ang_v = 0.0021
    results = 1
    G = -9.8
    shooting_time = 1
    t_separation = 0.051
    pred_samples_factor = 1
    just_last_sample = 0
    rtol_aux = 1e-5
    atol_aux = 1e-7
    datasets_vec_original = [training_dataset, 'fs_sim_rand3', 'fs_sim_rand4', 'fs_sim24', 'fs_sim30', 'fs_sim46', 'fs_sim10shots', 'fs_sim10shots2', 'fs_sim10shots5', 'fs_sim20shots2', 'fs_sim50shots', 'fs_sim50shots2', 'fs_sim100shots2'] # the training dataset
    plot_select_data_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # no selection of data
    plot_t_select_data_initial_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    plot_t_select_data_final_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    plot_t_u_flag_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # no guard
    plot_t_u_before_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    plot_t_u_after_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Looping through all of the different sets of datasets
    for suffic_idx in range(len(preffix_vec)):

        preffix = preffix_vec[suffic_idx]
        suffix = suffix_vec[suffic_idx]

        # Produce the final file names for the desired system
        datasets_vec = []
        aux_file_path = preffix + training_dataset + suffix
        file_path = "./data/" + folder_gimble + "/" + aux_file_path + ".txt"
        for k in range(len(datasets_vec_original)):
            aux = preffix + datasets_vec_original[k] + suffix
            datasets_vec.append(aux)
        
        directory = 'outputs/'+os.path.splitext(os.path.basename(file_path))[0]+'/mats_la'
        os.makedirs(directory, exist_ok=True)

        est = Estimator(file_path, t_u_before=t_u_before, t_u_after=t_u_after)
        theta = est.solve()

        # save results
        np.save(os.path.join(directory, "laa_theta.npy"), theta)
        np.save(os.path.join(directory, "laa_J1.npy"), est.J1)
        np.save(os.path.join(directory, "laa_c.npy"), est.c)
        np.save(os.path.join(directory, "laa_A1F.npy"), est.A1F)
        np.save(os.path.join(directory, "laa_A1M.npy"), est.A1M)
        print("Saved estimates to:", directory)

        print(f"J1:\n {est.J1}\n")
        print(f"c:\n {est.c}\n")
        print(f"A1F:\n {est.A1F}\n")
        print(f"A1M:\n {est.A1M}\n")

        # Get the true parameters
        true_ms_sim_A1M = np.load('./mats/ms_sim_A1M.npy')
        true_ms_sim_A1F = np.load('./mats/ms_sim_A1F.npy')
        true_ms_sim_c = np.load('./mats/ms_sim_c.npy')
        true_ms_sim_c = np.array(true_ms_sim_c)
        true_ms_sim_c = true_ms_sim_c.reshape(3, 1)
        true_ms_sim_J1 = np.load('./mats/ms_sim_J1.npy')
        true_ms_sim_L = np.linalg.cholesky(true_ms_sim_J1)
        true_ms_sim_L = true_ms_sim_L.T
        true_ms_sim_L = np.array([true_ms_sim_L[0][0],true_ms_sim_L[1][0],true_ms_sim_L[1][1],true_ms_sim_L[2][0],true_ms_sim_L[2][1],true_ms_sim_L[2][2]])
        true_initial_params_list = [true_ms_sim_L, true_ms_sim_c, true_ms_sim_A1M, true_ms_sim_A1F]
        true_initial_params = np.concatenate(
            [param.flatten() for param in true_initial_params_list]
        )
        true_params = torch.tensor(true_initial_params, requires_grad=True, dtype=torch.float32)
        # Parameters now computed
        ms_sim_L = np.linalg.cholesky(est.J1)
        ms_sim_L = np.array([ms_sim_L[0][0],ms_sim_L[1][0],ms_sim_L[1][1],ms_sim_L[2][0],ms_sim_L[2][1],ms_sim_L[2][2]])
        initial_params_list = [ms_sim_L, est.c, est.A1M, est.A1F]
        initial_params = np.concatenate([param.flatten() for param in initial_params_list])
        params = torch.tensor(initial_params, requires_grad=True, dtype=torch.float32)    
        # Compute quadratic differnece between true and estimated parameters
        quadratic_difference = torch.sum((true_params - params).pow(2)).item()
        print("quadratic_difference: ",quadratic_difference)


        if results:
            # ------------------------------------------------------- Plots with the final solution -------------------------------------------------------
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
            def f(t, x, params, u):
            # def f(t, x, params):
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

            # Define the loss function L
            def loss_fn(state, true_pos, true_vel, true_quat, true_ang_v, hparam_pos, hparam_vel, hparam_quat, hparam_ang_v):
                # Get the predicted quaternion from the state
                pred_pos, pred_vel, pred_quat, pred_ang_v = recover_state(state)

                # Dot Product Loss for the quaternions
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
            def shooting(params, t, t_np, positions, velocities, ang_v, quaternions, actuation, just_last_sample, rtol_aux, atol_aux, shooting_time, hparam_pos, hparam_vel, hparam_quat, hparam_ang_v, t_separation):

                # Define a wrapper function that sums up the system dynamics
                def f_wrapper(t, x, params, u):
                    return f(t, x, params, u)
                
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
                            lambda t, x: f_wrapper(t, x, params, actuation_value),
                            # f_wrapper,
                            state,
                            t[start_sample_chunk_i:end_sample_chunk_i_aux+1].float(), # t from index start_sample_chunk_i to end_sample_chunk_i_aux (needs the +1)
                            rtol=rtol_aux,  # Relative tolerance
                            atol=atol_aux,  # Absolute tolerance
                            method='dopri5'
                        )
                        end_odeint_time = time.time()
                        time_odeint.append(end_odeint_time - start_odeint_time)

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

            # Function to change the final vectors and avoid connecting points to far away in the final graphs
            def break_plot_on_gaps(t, y, max_gap):
                t = np.asarray(t)
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
            def plot_shooting_results_rmse(datasets_vec, plot_select_data_vec, plot_t_select_data_initial_vec, plot_t_select_data_final_vec, plot_t_u_flag_vec, plot_t_u_before_vec, plot_t_u_after_vec, params, hparam_pos, hparam_vel, hparam_quat, hparam_ang_v, pred_samples_factor, shooting_time, just_last_sample, rtol_aux, atol_aux, file, directory1, directory2, directory3, str_t_u_flag, folder_gimble, lin_test_mode):

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
                        t, positions, velocities, ang_v, quaternions, actuation = select_data_range(t, positions, velocities, ang_v, quaternions, actuation, plot_t_select_data_initial_vec[i], plot_t_select_data_final_vec[i])

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

                    # Get the true Euler angles
                    euler_angles = quaternion_to_euler_angles(quaternions)
                    phi_angles = [item[0] for item in euler_angles]
                    theta_angles = [item[1] for item in euler_angles]
                    psi_angles = [item[2] for item in euler_angles]
                    
                    # Shooting
                    loss,pred_positions,pred_velocities,pred_quaternions,pred_ang_v,_ = shooting(params, t, t_np, positions, velocities, ang_v, quaternions, actuation, just_last_sample, rtol_aux, atol_aux, shooting_time, hparam_pos, hparam_vel, hparam_quat, hparam_ang_v, t_separation)
                    mean_loss = (loss.item())/data_size
                    file.write(f"Total loss (hparam_pos={plot_hparam_pos},hparam_vel={plot_hparam_vel},hparam_quat={plot_hparam_quat},hparam_ang_v={plot_hparam_ang_v}): {loss.item()}\n")
                    file.write(f"Mean Total loss (hparam_pos={plot_hparam_pos},hparam_vel={plot_hparam_vel},hparam_quat={plot_hparam_quat},hparam_ang_v={plot_hparam_ang_v}): {mean_loss}\n")

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
                    aux_solver = "laa_results_"
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
                    else: #Test datasets performance
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

                if lin_test_mode == 0:
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
                elif lin_test_mode == 1:
                    mean_test_pos_rmse_vec.append(mean_training_pos_rmse)
                    mean_test_vel_rmse_vec.append(mean_training_vel_rmse)
                    mean_test_att_rmse_vec.append(mean_training_att_rmse)
                    mean_test_ang_v_rmse_vec.append(mean_training_ang_v_rmse)
                    file.write(f"\nNon linear Test datasets mean performances:\n")
                    file.write(f"mean positions rmse: {np.mean(mean_test_pos_rmse_vec)}\n")
                    file.write(f"mean velocities rmse: {np.mean(mean_test_vel_rmse_vec)}\n")
                    file.write(f"mean attitudes rmse: {np.mean(mean_test_att_rmse_vec)}\n")
                    file.write(f"mean ang_v rmse: {np.mean(mean_test_ang_v_rmse_vec)}\n")


            ms_sim_L = np.linalg.cholesky(est.J1)
            ms_sim_L = np.array([ms_sim_L[0][0],ms_sim_L[1][0],ms_sim_L[1][1],ms_sim_L[2][0],ms_sim_L[2][1],ms_sim_L[2][2]])
            initial_params_list = [ms_sim_L, est.c, est.A1M, est.A1F]
            initial_params = np.concatenate([param.flatten() for param in initial_params_list])
            params = torch.tensor(initial_params, requires_grad=True, dtype=torch.float32)  

            # Creating the results file and directory for the plots
            directory = 'outputs/'+os.path.splitext(os.path.basename(file_path))[0]
            os.makedirs(directory, exist_ok=True)
            file_name = directory + '/results.txt'
            file = open(file_name, "w")
            directory1 = directory + '/shooting_plots_png'
            directory2 = directory + '/shooting_plots_pdf'
            directory3 = directory + '/shooting_plots_data'
            os.makedirs(directory1, exist_ok=True)
            os.makedirs(directory2, exist_ok=True)
            os.makedirs(directory3, exist_ok=True)
            file.write(f"Squared difference of the final solution to the ground truth: {quadratic_difference}\n\n")
            file.write("test sets: " + str(datasets_vec[1:]) + "\n\n")
            file.write("Shooting details to obtain performances:\n")
            file.write("training dataset file_path: " + str(file_path) + "\n")
            file.write("plot_hparam_pos: " + str(plot_hparam_pos) + "\n")
            file.write("plot_hparam_vel: " + str(plot_hparam_vel) + "\n")
            file.write("plot_hparam_quat: " + str(plot_hparam_quat) + "\n")
            file.write("plot_hparam_ang_v: " + str(plot_hparam_ang_v) + "\n")
            file.write("just_last_sample: " + str(just_last_sample) + "\n")
            file.write("rtol_aux: " + str(rtol_aux) + "\n")
            file.write("atol_aux: " + str(atol_aux) + "\n")
            file.write("shooting_time: " + str(shooting_time) + "\n")
            file.write("pred_samples_factor: " + str(pred_samples_factor) + "\n")
            file.write("t_separation: " + str(t_separation) + "\n")
            file.write("t_u_before: " + str(t_u_before) + "\n")
            file.write("t_u_after: " + str(t_u_after) + "\n\n")

            hparam_pos = plot_hparam_pos
            hparam_vel = plot_hparam_vel
            hparam_quat = plot_hparam_quat
            hparam_ang_v = plot_hparam_ang_v            

            # Output string for the removed interval around each actuaction change
            if t_u_flag:
                str_t_u_flag = '_tu'+str(t_u_flag)+'b'+str(t_u_before)+'a'+str(t_u_after)
            else:
                str_t_u_flag = '_tu'+str(t_u_flag)

            lin_test_mode = 0

            # Plots the shooting results and compute RMSE results
            plot_shooting_results_rmse(datasets_vec, plot_select_data_vec, plot_t_select_data_initial_vec, plot_t_select_data_final_vec, plot_t_u_flag_vec, plot_t_u_before_vec, plot_t_u_after_vec, params, hparam_pos, hparam_vel, hparam_quat, hparam_ang_v, pred_samples_factor, shooting_time, just_last_sample, rtol_aux, atol_aux, file, directory1, directory2, directory3, str_t_u_flag, folder_gimble, lin_test_mode)

            # Close the report .txt file
            file.close()

            # Computes the performances and shooting with the test datasets with 
            if evaluate_on_linear == 1:
                # Creating the results file and directory for the plots
                directory = 'outputs/'+os.path.splitext(os.path.basename(file_path))[0]
                os.makedirs(directory, exist_ok=True)
                file_name = directory + '/results_lin_test_sets.txt'
                file = open(file_name, "w")
                directory1 = directory + '/shooting_plots_png_lin_test_sets'
                directory2 = directory + '/shooting_plots_pdf_lin_test_sets'
                directory3 = directory + '/shooting_plots_data_lin_test_sets'
                os.makedirs(directory1, exist_ok=True)
                os.makedirs(directory2, exist_ok=True)
                os.makedirs(directory3, exist_ok=True)

                datasets_vec_lin_test_sets = datasets_vec_original[1:]
                file.write("test sets: " + str(datasets_vec_lin_test_sets) + "\n\n")

                lin_test_mode = 1
                # Plots the shooting results and compute RMSE results
                plot_shooting_results_rmse(datasets_vec_lin_test_sets, plot_select_data_vec, plot_t_select_data_initial_vec, plot_t_select_data_final_vec, plot_t_u_flag_vec, plot_t_u_before_vec, plot_t_u_after_vec, params, hparam_pos, hparam_vel, hparam_quat, hparam_ang_v , pred_samples_factor, shooting_time, just_last_sample, rtol_aux, atol_aux, file, directory1, directory2, directory3, str_t_u_flag, folder_gimble, lin_test_mode)

                # Close the report .txt file
                file.close()

if __name__ == "__main__":
    main(sys.argv)
