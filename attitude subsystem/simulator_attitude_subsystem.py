import numpy as np
import numpy.linalg as la
import scipy.integrate as intgr
import scipy.spatial.transform as trf
import matplotlib.pyplot as plt
import io
import os
import sys

def S(v):
    """Skew-symmetric matrix"""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

class Simulator:
    m = 1
    r = 0.25
    h = 0.15
    J = np.diag([m * (3 * r * r + h * h) / 12, m * (3 * r * r + h * h) / 12, m * r * r / 2])
    Jinv = la.inv(J)
    c = np.array([0.01, 0.02, 0.05])
    g = np.array([0, 0, -9.8])
    mu = 0.00

    Atxt = """
0.000000000000000000e+00 -7.094064799162224100e-01 7.094064799162225210e-01 -1.003171929053524623e-16 -7.094064799162221879e-01 7.094064799162224100e-01
-8.191520442889916875e-01 4.095760221444959548e-01 4.095760221444956772e-01 -8.191520442889916875e-01 4.095760221444962323e-01 4.095760221444958438e-01
5.735764363510461594e-01 5.735764363510461594e-01 5.735764363510461594e-01 5.735764363510461594e-01 5.735764363510461594e-01 5.735764363510461594e-01
0.000000000000000000e+00 8.657114718190686564e-02 8.657114718190689340e-02 1.224202867855199215e-17 -8.657114718190685176e-02 -8.657114718190686564e-02
-9.996375025905729350e-02 -4.998187512952866063e-02 4.998187512952862593e-02 9.996375025905729350e-02 4.998187512952868838e-02 -4.998187512952864675e-02
-1.253285627227282151e-01 1.253285627227282151e-01 -1.253285627227282151e-01 1.253285627227282151e-01 -1.253285627227282151e-01 1.253285627227282151e-01
"""
    A = np.loadtxt(io.StringIO(Atxt))
    M = np.block([[m * np.eye(3), -m * S(c)], [m * S(c), J]])
    Minv = la.inv(M)

    def dyn_static(self, state, u, non_linear_effects, hparam_nlinear, B_M, phi_funcs, CM_list, hparam_nlinear_drag, gimble):           
        (x, v, R, w) = state
        if non_linear_effects == 0:
            M = self.A[3:6, :] @ u
        # Quadratic Model
        if non_linear_effects == 1:
            M = self.A[3:6, :] @ u + hparam_nlinear * (B_M @ np.outer(u, u).reshape(-1))
        # Basis Function Expansion
        if non_linear_effects == 2:
            M = self.A[3:6, :] @ u
            # nonlinear contributions
            for k, phi_k in enumerate(phi_funcs):
                M += hparam_nlinear * (CM_list[k] @ phi_k(u) )
        # Air Drag
        if non_linear_effects == 3:
            B_drag = np.ones((3, 9))
            M = self.A[3:6, :] @ u + hparam_nlinear_drag * (B_drag @ np.outer(w, w).reshape(-1))
        # QM and AD
        if non_linear_effects == 4:
            B_drag = np.ones((3, 9))
            M = self.A[3:6, :] @ u + hparam_nlinear * (B_M @ np.outer(u, u).reshape(-1)) + hparam_nlinear_drag * (B_drag @ np.outer(w, w).reshape(-1))
        # BFE and AD
        if non_linear_effects == 5:
            B_drag = np.ones((3, 9))
            M = self.A[3:6, :] @ u + hparam_nlinear_drag * (B_drag @ np.outer(w, w).reshape(-1))
            # nonlinear contributions
            for k, phi_k in enumerate(phi_funcs):
                M += hparam_nlinear * (CM_list[k] @ phi_k(u) )
        # QM, BFE and AD
        if non_linear_effects == 6:
            B_drag = np.ones((3, 9))
            M = self.A[3:6, :] @ u + hparam_nlinear * (B_M @ np.outer(u, u).reshape(-1)) + hparam_nlinear_drag * (B_drag @ np.outer(w, w).reshape(-1))
            # nonlinear contributions
            for k, phi_k in enumerate(phi_funcs):
                M += hparam_nlinear * (CM_list[k] @ phi_k(u) )


        C = np.hstack([self.m * S(w) @ S(w) @ self.c, S(w) @ self.J @ w])
        if gimble:
            q_ddot = np.hstack([
                np.zeros(3),
                self.Jinv @ (M + self.m * S(self.c) @ R.T @ self.g - C[3:6])
            ])
        else:
            q_ddot = np.hstack([
                np.zeros(3),
                self.Jinv @ (M - C[3:6])
            ])
        x_dot = v
        v_dot = np.zeros(3)
        R_dot = R @ S(w)
        w_dot = q_ddot[3:6]
        return (x_dot, v_dot, R_dot, w_dot)

    dyn = dyn_static

    def ode(self, s0, u, delta_t, non_linear_effects, hparam_nlinear, B_M, phi_funcs, CM_list, hparam_nlinear_drag, gimble):
        def flat(s):
            (x, v, R, w) = s
            return np.hstack([x, v, R.flatten(), w])

        def unflat(y):
            x = y[0:3]
            v = y[3:6]
            R = y[6:15].reshape(3, 3)
            w = y[15:18]
            return (x, v, R, w)

        def f(t, y):
            s = unflat(y)
            s_dot = self.dyn(s, u, non_linear_effects, hparam_nlinear, B_M, phi_funcs, CM_list, hparam_nlinear_drag, gimble)
            return flat(s_dot)

        y0 = flat(s0)
        ret = intgr.solve_ivp(f, (0, delta_t), y0)
        return unflat(ret.y[:, -1])

def generate_data(actuations, u_repetition, initial, time_offset, non_linear_effects, hparam_nlinear, B_M, phi_funcs, CM_list, hparam_nlinear_drag, gimble, append=False):
    sim = Simulator()

    u = np.vstack([np.tile(actuations[i, :], (u_repetition, 1)) for i in range(len(actuations))]).T

    traj = [initial]
    s = initial

    print("Simulation time (seconds):", u.shape[1] * T)

    for i in range(u.shape[1]):
        s = sim.ode(s, u[:, i], T, non_linear_effects, hparam_nlinear, B_M, phi_funcs, CM_list, hparam_nlinear_drag, gimble)
        traj.append(s)

    xs = np.vstack([s[0] for s in traj])
    vs = np.vstack([s[1] for s in traj])
    Rs = np.array([s[2] for s in traj])
    ws = np.vstack([s[3] for s in traj])

    timestamps = [time_offset + (T * i) + T for i in range(u.shape[1])]
    return timestamps, ws, Rs, u

def write_to_file_static_txt(timestamp, w, R, u, output_file, append=False):
    u = u.T
    quaternion = trf.Rotation.from_matrix(R).as_quat()
    os.makedirs("simulator_outputs", exist_ok=True)

    mode = "a" if append else "w"
    output_file_path = "simulator_outputs/"+output_file+".txt"
    with open(output_file_path, mode) as fp:
        for i in range(len(timestamp)):
            fp.write(f"{timestamp[i]}\n")
            fp.write(f"{w[i][0]} {w[i][1]} {w[i][2]}\n")
            fp.write(f"{quaternion[i][0]} {quaternion[i][1]} {quaternion[i][2]} {quaternion[i][3]}\n")
            fp.write(f"{u[i][0]} {u[i][1]} {u[i][2]} {u[i][3]} {u[i][4]} {u[i][5]}\n\n")

def quaternion_to_rotation_matrix(q):
    q1, q2, q3, q4 = q
    R = np.array([
        [1 - 2*(q2**2 + q3**2),     2*(q1*q2 - q3*q4),     2*(q1*q3 + q2*q4)],
        [2*(q1*q2 + q3*q4),         1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q1*q4)],
        [2*(q1*q3 - q2*q4),         2*(q2*q3 + q1*q4),     1 - 2*(q1**2 + q2**2)]
    ])
    return R

def sample_random_rotation_matrix():
    u1, u2, u3 = np.random.uniform(0, 1, 3)
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    q = np.array([q1, q2, q3, q4])
    return quaternion_to_rotation_matrix(q)

def sample_small_rotation(max_angle=0.2):
    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.random.uniform(-max_angle, max_angle)
    qw = np.cos(angle/2)
    qx, qy, qz = axis * np.sin(angle/2)
    q = np.array([qw, qx, qy, qz])
    return quaternion_to_rotation_matrix(q)  # <- now 3x3


if __name__ == "__main__":

    # Quadratic Model for non linear effects
    # Generate random Quadratic Model B matrix
    B_M = np.random.randn(3, 36)
    B_M /= np.linalg.norm(B_M) # normalize them so non linear effects don't blow up
    print("B_M: ",B_M)
    print("B_M.shape: ",B_M.shape)
    print("np.linalg.norm(B_M): ",np.linalg.norm(B_M))
    
    # Basis Function Expansion for non linear effects
    # Define basis functions
    def phi1(u): return np.sin(u)     # shape (6,)
    def phi2(u): return np.tanh(u)    # shape (6,)
    def phi3(u): return u**2          # shape (6,)
    phi_funcs = [phi1, phi2, phi3]
    # Generate random Quadratic Model C matrices
    CM_list = [np.random.randn(3, 6) for _ in phi_funcs]
    for k in range(len(phi_funcs)):
        CM_list[k] /= np.linalg.norm(CM_list[k])
        print(f"CM_list[{k}]: ",CM_list[k])
        print(f"CM_list[{k}].shape: ",CM_list[k].shape)
        print(f"np.linalg.norm(CM_list[{k}]): ",np.linalg.norm(CM_list[k]))

    unitary_actuations = np.array([
                        [1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1],
                    ])
    
    gimble = 0 # 0: no gimble, 1: gimble
    T = 1 / 50 #0.02s of sample time
    time_diff_next_shoot = 1 # time difference between the end of a shoot and the beginning of the next one
    print_actuations = 0

    # maximum values of angular velocity in rad/s and actuation
    max_w = 3

    # Infos:
    # non_linear_effects:
    # 0: no non linear effects, 1: Quadratic Model (QM), 2: Basis Function Expansion (BFE); 3: Air Drag (AD); 
    # 4: QM and AD; 5: BFE and AD; 6: QM, BFE and AD
    # u_vec_time: duration of each actuation in seconds
    # num_shots: number of different shoots with different initializations
    # actuation_mode:
    # actuation_mode = 0: unitary actuations
    # actuation_mode = 24: close-to-unitary actuations 1
    # actuation_mode = 30: close-to-unitary actuations 2
    # actuation_mode = 46: close-to-unitary actuations 3
    # actuation_mode = 1: random actuations
    # actuation_mode = 2: random actuations and initialization state in multiple individual chunks
    # actuation_mode = 3: random unitary actuations

    # Dataset parameters for Only One dataset:
    # non_linear_effects_vec = [0]
    # hparam_nlinear_vec = [0]
    # hparam_nlinear_drag_vec = [0]
    # output_name_suffix_vec = [""]
    # # -------------------
    # actuation_mode_vec = [0]
    # initialization_random_vec = [0]
    # u_vec_time_vec = [2]
    # num_shots_vec = [1]
    # max_u_vec = [0]
    # output_file_vec_original = ["sim_first"]

    # Dataset parameters for Multiple Datasets of one same system
    non_linear_effects_vec = [0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 4, 4, 5, 6, 6]
    hparam_nlinear_vec = [0, 0.05, 0.1, 0.2, 0.4, 1, 2, 5, 0.05, 0.1, 0.2, 0.4, 1, 0, 0, 0.2, 0.4, 0.2, 0.2, 0.35]
    hparam_nlinear_drag_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0001, 0.0002, 0.0001, 0.0002, 0.0001, 0.0001, 0.0001]
    output_name_suffix_vec = ["", "QM2", "QM3", "QM4", "QM6", "QM7", "QM8", "QM9", "BFE1", "BFE2", "BFE3", "BFE4", "BFE5", "AD1", "AD2", "QM_AD1", "QM_AD2", "BFE_AD1", "QM_BFE_AD1", "QM_BFE_AD2"]
    # -------------------
    actuation_mode_vec = [0, 24, 30, 46, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    initialization_random_vec = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    u_vec_time_vec = [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    num_shots_vec = [1, 1, 1, 1, 1, 1, 10, 10, 10, 10, 10, 20, 20, 50, 50, 100, 100, 500, 10, 20, 46, 102]
    max_u_vec = [0, 0, 0, 0, 0.2, 0.2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    output_file_vec_original = ["sim_first", "sim24", "sim30", "sim46", "sim_rand3", "sim_rand4", "sim10shots", "sim10shots2", "sim10shots3", "sim10shots4", "sim10shots5", "sim20shots", "sim20shots2", "sim50shots", "sim50shots2", "sim100shots", "sim100shots2", "sim500shots", "sim16shots_first", "sim32shots_first", "sim64shots_first", "sim128shots_first"]




    # Looping through all the different systems
    for j in range(len(non_linear_effects_vec)):

        non_linear_effects = non_linear_effects_vec[j]
        hparam_nlinear = hparam_nlinear_vec[j]
        hparam_nlinear_drag = hparam_nlinear_drag_vec[j]

        # Adding preffix and suffix if we are adding non linear perturbations
        if non_linear_effects == 0:
            output_file_vec = output_file_vec_original
        else:
            output_file_vec = []
            for k in range(len(output_file_vec_original)):
                aux = "simnl_" + output_file_vec_original[k] + "_" + output_name_suffix_vec[j]
                output_file_vec.append(aux)

        # Looping through all the different datasets
        for k in range(len(actuation_mode_vec)):

            actuation_mode = actuation_mode_vec[k]
            initialization_random = initialization_random_vec[k]
            u_vec_time = u_vec_time_vec[k]
            num_shots = num_shots_vec[k]
            max_u = max_u_vec[k]
            output_file = output_file_vec[k]

            u_vec_time_dataset_vec = []
            actuations_vec = []
            initial_vec = []

            # Generate actuations and initial states for each individual long shot
            for i in range(num_shots):
                u_vec_time_dataset_vec.append(u_vec_time)
                # ----- to change -----
                if actuation_mode == 0:
                    actuations = unitary_actuations
                if actuation_mode == 24:
                    actuations = np.array([
                                    [1, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0.5, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1],
                                ])
                if actuation_mode == 30:
                    actuations = np.array([
                                    [0.5, 0, 0, 0, 0, 0],
                                    [0, 0.5, 0, 0, 0, 0],
                                    [0, 0, 0.5, 0, 0, 0],
                                    [0, 0, 0, 0.5, 0, 0],
                                    [0, 0, 0, 0, 0.5, 0],
                                    [0, 0, 0, 0, 0, 0.5],
                                ])
                if actuation_mode == 46:
                    actuations = np.array([
                                    [-0.5, 0, 0, 0, 0, 0],
                                    [0, -0.5, 0, 0, 0, 0],
                                    [0, 0, -0.5, 0, 0, 0],
                                    [0, 0, 0, -0.5, 0, 0],
                                    [0, 0, 0, 0, -0.5, 0],
                                    [0, 0, 0, 0, 0, -0.2],
                                ])
                if actuation_mode == 1: 
                    actuations = np.random.uniform(-max_u, max_u, size=(6, 6))
                if actuation_mode == 2:
                    actuation = np.random.uniform(-max_u, max_u, size=6)
                    actuations = np.array([actuation])
                if actuation_mode == 3:
                    actuation = unitary_actuations[i % len(unitary_actuations)]

                if print_actuations == 1:
                    print("actuations: ",actuations)
                actuations_vec.append(actuations)

                if initialization_random == 0:
                    rotation = np.eye(3)  # Use identity matrix for initial rotation
                    angular_velocity = np.zeros(3)  # Use zero angular velocity for initial state
                if initialization_random == 1:
                    rotation = sample_random_rotation_matrix()
                    # rotation = sample_small_rotation()
                    angular_velocity = np.random.uniform(-max_w, max_w, size=3)
                # ------------------
                position = np.zeros(3) #not using right now
                linear_velocity = np.zeros(3) #not using right now
                initial = (position, linear_velocity, rotation, angular_velocity)
                initial_vec.append(initial)

            time_offset = 0.0
            # Generate data for each long shot
            for i in range(num_shots):

                u_vec_time = u_vec_time_dataset_vec[i]
                actuations = actuations_vec[i]
                initial = initial_vec[i]

                u_repetition = int(u_vec_time/T)
                timestamps, ws, Rs, u = generate_data(actuations, u_repetition, initial, time_offset, non_linear_effects, hparam_nlinear, B_M, phi_funcs, CM_list, hparam_nlinear_drag, gimble)
                
                write_to_file_static_txt(timestamps, ws, Rs, u, output_file, append=(i != 0))

                # Update time_offset for next sequence (+1s after last sample)
                time_offset = timestamps[-1] + time_diff_next_shoot

            print(f"Dataset {output_file_vec[k]} finished and saved!")
        
        # Final print
        print(f"{j+1}/{len(non_linear_effects_vec)} systems finished!\n")

    # Final print
    print(f"All finished!")

    

