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

    def dyn_full(self, state, u, non_linear_effects, hparam_nlinear, B_M, B_F, phi_funcs, CM_list, CF_list):
        """Continuous time dynamics with explicit rotation matrix"""
        (x, v, R, w) = state
        if non_linear_effects == 0:
            F = R @ self.A[0:3, :] @ u
            M = self.A[3:6, :] @ u
        # Quadratic Model
        if non_linear_effects == 1:
            F = R @ self.A[0:3, :] @ u + hparam_nlinear * (B_F @ np.outer(u, u).reshape(-1))
            M = self.A[3:6, :] @ u + hparam_nlinear * (B_M @ np.outer(u, u).reshape(-1))
        # Basis Function Expansion
        if non_linear_effects == 2:
            F = R @ self.A[0:3, :] @ u
            M = self.A[3:6, :] @ u
            # nonlinear contributions
            for k, phi_k in enumerate(phi_funcs):
                F += hparam_nlinear * (CF_list[k] @ phi_k(u) )
                M += hparam_nlinear * (CM_list[k] @ phi_k(u) )
        # Air Drag
        if non_linear_effects == 3:
            B_drag = np.ones((3, 9))
            F = R @ self.A[0:3, :] @ u + hparam_nlinear_drag * (B_drag @ np.outer(v, v).reshape(-1))
            M = self.A[3:6, :] @ u + hparam_nlinear_drag * (B_drag @ np.outer(w, w).reshape(-1))
        # QM and AD
        if non_linear_effects == 4:
            B_drag = np.ones((3, 9))
            F = R @ self.A[0:3, :] @ u + hparam_nlinear * (B_F @ np.outer(u, u).reshape(-1)) + hparam_nlinear_drag * (B_drag @ np.outer(v, v).reshape(-1))
            M = self.A[3:6, :] @ u + hparam_nlinear * (B_M @ np.outer(u, u).reshape(-1)) + hparam_nlinear_drag * (B_drag @ np.outer(w, w).reshape(-1))
        # BFE and AD
        if non_linear_effects == 5:
            B_drag = np.ones((3, 9))
            F = R @ self.A[0:3, :] @ u + hparam_nlinear_drag * (B_drag @ np.outer(v, v).reshape(-1))
            M = self.A[3:6, :] @ u + hparam_nlinear_drag * (B_drag @ np.outer(w, w).reshape(-1))
            # nonlinear contributions
            for k, phi_k in enumerate(phi_funcs):
                F += hparam_nlinear * (CF_list[k] @ phi_k(u) )
                M += hparam_nlinear * (CM_list[k] @ phi_k(u) )
        # QM, BFE and AD
        if non_linear_effects == 6:
            B_drag = np.ones((3, 9))
            F = R @ self.A[0:3, :] @ u + hparam_nlinear * (B_F @ np.outer(u, u).reshape(-1)) + hparam_nlinear_drag * (B_drag @ np.outer(v, v).reshape(-1))
            M = self.A[3:6, :] @ u + hparam_nlinear * (B_M @ np.outer(u, u).reshape(-1)) + hparam_nlinear_drag * (B_drag @ np.outer(w, w).reshape(-1))
            # nonlinear contributions
            for k, phi_k in enumerate(phi_funcs):
                F += hparam_nlinear * (CF_list[k] @ phi_k(u) )
                M += hparam_nlinear * (CM_list[k] @ phi_k(u) )

        # Mass matrix
        Mmat = np.block([
            [self.m * np.eye(3), -self.m * R @ S(self.c)],
            [np.zeros((3,3)), self.J]
        ])
        # Nonlinear terms (Coriolis/centrifugal)
        Cvec = np.hstack([
            self.m * R @ (S(w) @ S(w) @ self.c),
            S(w) @ self.J @ w
        ])
        # RHS (forces)
        RHS = np.hstack([
            F,
            M
        ])
        # Solve for accelerations [v_dot, w_dot]
        q_ddot = np.linalg.solve(Mmat, RHS - Cvec)

        v_dot = q_ddot[0:3]
        w_dot = q_ddot[3:6]

        # Kinematics
        x_dot = v # world-frame velocity of CoM
        R_dot = R @ S(w)

        return (x_dot, v_dot, R_dot, w_dot)

    dyn = dyn_full

    def ode(self, s0, u, delta_t, non_linear_effects, hparam_nlinear, B_M, B_F, phi_funcs, CM_list, CF_list):
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
            s_dot = self.dyn(s, u, non_linear_effects, hparam_nlinear, B_M, B_F, phi_funcs, CM_list, CF_list)
            return flat(s_dot)

        y0 = flat(s0)
        ret = intgr.solve_ivp(f, (0, delta_t), y0)
        return unflat(ret.y[:, -1])

def generate_data(actuations, u_repetition, initial, time_offset, non_linear_effects, hparam_nlinear, B_M, B_F, phi_funcs, CM_list, CF_list, append=False):
    sim = Simulator()

    u = np.vstack([np.tile(actuations[i, :], (u_repetition, 1)) for i in range(len(actuations))]).T

    traj = [initial]
    s = initial

    print("Simulation time (seconds):", u.shape[1] * T)

    for i in range(u.shape[1]):
        s = sim.ode(s, u[:, i], T, non_linear_effects, hparam_nlinear, B_M, B_F, phi_funcs, CM_list, CF_list)
        traj.append(s)

    xs = np.vstack([s[0] for s in traj])
    vs = np.vstack([s[1] for s in traj])
    Rs = np.array([s[2] for s in traj])
    ws = np.vstack([s[3] for s in traj])

    timestamps = [time_offset + (T * i) + T for i in range(u.shape[1])]
    return timestamps, xs, vs, ws, Rs, u

def write_to_file_static_txt(timestamps, xs, vs, ws, Rs, u, output_file, append=False):
    u = u.T
    quaternions = trf.Rotation.from_matrix(Rs).as_quat()
    os.makedirs("simulator_outputs", exist_ok=True)

    mode = "a" if append else "w"
    output_file_path = "simulator_outputs/"+output_file+".txt"
    with open(output_file_path, mode) as fp:
        for i in range(len(timestamps)):
            fp.write(f"{timestamps[i]}\n")
            fp.write(f"{xs[i][0]} {xs[i][1]} {xs[i][2]}\n")
            fp.write(f"{vs[i][0]} {vs[i][1]} {vs[i][2]}\n")
            fp.write(f"{ws[i][0]} {ws[i][1]} {ws[i][2]}\n")
            fp.write(f"{quaternions[i][0]} {quaternions[i][1]} {quaternions[i][2]} {quaternions[i][3]}\n")
            fp.write(" ".join(str(val) for val in u[i]) + "\n\n")

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
    # Generate random Quadratic Model B matrix for the Force
    B_F = np.random.randn(3, 36)
    B_F /= np.linalg.norm(B_F) # normalize them so non linear effects don't blow up
    print(f"B_F: ",B_F)

    # Generate random Quadratic Model B matrix for the Torque
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

    # Generate random Quadratic Model C matrices for the Force
    CF_list = [np.random.randn(3, 6) for _ in phi_funcs]
    for k in range(len(phi_funcs)):
        CF_list[k] /= np.linalg.norm(CF_list[k])
        print(f"CF_list[{k}]: ",CF_list[k])

    # Generate random Quadratic Model C matrices for the Torque
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

    T = 1 / 50 #0.02s of sample time
    time_diff_next_shoot = 1 # time difference between the end of a shoot and the beginning of the next one
    print_actuations = 0

    # maximum values of angular velocity in rad/s and actuation
    max_p = 3
    max_v = 2
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
    # output_file_vec_original = ["fs_sim_first"]
    # add_sim_first_end_vec = [0]
    # u_vec_time_sim_first_vec = [0]

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
    output_file_vec_original = ["fs_sim_first", "fs_sim24", "fs_sim30", "fs_sim46", "fs_sim_rand3", "fs_sim_rand4", "fs_sim10shots", "fs_sim10shots2", "fs_sim10shots3", "fs_sim10shots4", "fs_sim10shots5", "fs_sim20shots", "fs_sim20shots2", "fs_sim50shots", "fs_sim50shots2", "fs_sim100shots", "fs_sim100shots2", "fs_sim500shots", "fs_sim16shots_first", "fs_sim32shots_first", "fs_sim64shots_first", "fs_sim128shots_first"]
    add_sim_first_end_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    u_vec_time_sim_first_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4]




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
            add_sim_first_end = add_sim_first_end_vec[k]
            u_vec_time_sim_first = u_vec_time_sim_first_vec[k]

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
                    position = np.zeros(3)
                    linear_velocity = np.zeros(3)
                if initialization_random == 1:
                    rotation = sample_random_rotation_matrix()
                    # rotation = sample_small_rotation()
                    angular_velocity = np.random.uniform(-max_w, max_w, size=3)
                    position = np.random.uniform(-max_p, max_p, size=3)
                    linear_velocity = np.random.uniform(-max_v, max_v, size=3)
                # ------------------
                initial = (position, linear_velocity, rotation, angular_velocity)
                initial_vec.append(initial)

            # Generate actuations and initial states for the added sim_first
            if add_sim_first_end == 1:
                actuations_end = unitary_actuations
                if print_actuations == 1:
                    print("actuations_end: ",actuations_end)
                rotation = np.eye(3)  # Use identity matrix for initial rotation
                angular_velocity = np.zeros(3)  # Use zero angular velocity for initial state
                position = np.zeros(3)
                linear_velocity = np.zeros(3)
                initial_end = (position, linear_velocity, rotation, angular_velocity)

            time_offset = 0.0
            # Generate data for each long shot
            for i in range(num_shots):

                u_vec_time = u_vec_time_dataset_vec[i]
                actuations = actuations_vec[i]
                initial = initial_vec[i]

                u_repetition = int(u_vec_time/T)
                timestamps, xs, vs, ws, Rs, u = generate_data(actuations, u_repetition, initial, time_offset, non_linear_effects, hparam_nlinear, B_M, B_F, phi_funcs, CM_list, CF_list)

                write_to_file_static_txt(timestamps, xs, vs, ws, Rs, u, output_file, append=(i != 0))

                # Update time_offset for next sequence (+1s after last sample)
                time_offset = timestamps[-1] + time_diff_next_shoot
            
            # Add the sim_first at the end
            if add_sim_first_end == 1:
                u_vec_time = u_vec_time_sim_first
                actuations = actuations_end
                initial = initial_end

                u_repetition = int(u_vec_time/T)
                timestamps, xs, vs, ws, Rs, u = generate_data(actuations, u_repetition, initial, time_offset, non_linear_effects, hparam_nlinear, B_M, B_F, phi_funcs, CM_list, CF_list)

                write_to_file_static_txt(timestamps, xs, vs, ws, Rs, u, output_file, append=True)

            print(f"Dataset {output_file_vec[k]} finished and saved!")
        
        # Final print
        print(f"{j+1}/{len(non_linear_effects_vec)} systems finished!\n")

    # Final print
    print(f"All finished!")