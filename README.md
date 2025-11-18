# Modeling of Orbital Robotic Systems Using a Hybrid Model-Based and Data-Driven Approach
This repository contains the main code used in the master thesis "Modeling of Orbital Robotic Systems Using a Hybrid Model-Based and Data-Driven Approach" at Instituto Superior Técnico.

## Overview
This code was used in a master thesis project focused on modeling an orbital robot system using model-based, data-driven and hybrid model learning approaches and comparing its results. With validated system dynamics, we simulate robot trajectories and save them to then feed the different types of learning approaches, whose results were then compared. Both the simulator and the model learning scripts are available for the simplified attitude subsystem and for the full system of the robot, which includes both rotational and translational dynamics.

## Features
- **3D Rigid Body Simulation**: Implements a physics-based model for state estimation with the validated rigid-body robot equations, and simulates various trajectory types according to user specifications.
- **Results and Visualization**: Every robot model learning script automatically generates trajectory plots for a set of prediction scenarios defined by the user, as well as the perfomance RMSE shooting metrics for every state variable, enabling systematic performance evaluation and comparison.

## Usage
### Running the system simulators
The `simulator_full_system.py` script simulates robot trajetories considering the full system and saves them in `.txt` fyles with the following format:

timestamp
x_x x_y x_z
v_x v_y v_z
w_tk_x w_tk_y w_tk_z 
bi cj dk a
u_tk[0] u_tk[1] u_tk[2] u_tk[3] u_tk[4] u_tk[5]

Each entry corresponds to a state sample containing, respectively:
- sample timestamp,  
- position,  
- linear velocity,  
- angular velocity,  
- attitude expressed as quaternions,  
- and actuation forces.

Similarly, the `simulator_attitude_subsystem.py` script simulates robot trajetories considering the simplified attitude subsystem and saves them in `.txt` fyles with the following format:

timestamp
w_tk_x w_tk_y w_tk_z 
bi cj dk a
u_tk[0] u_tk[1] u_tk[2] u_tk[3] u_tk[4] u_tk[5].

To simulate a robot trajectory with predefined chosen options we run:
```sh
python simulator_full_system.py
python simulator_attitude_subsystem.py
```

### Running the robot model learning scripts
The `full_sys_mb_ode.py` and `att_subsys_mb_ode.py` consist of the model-based ODE approach learning scripts and the scripts `full_sys_sdp.py` and `att_subsys_sdp.py` consist of the model-based SPD approach learning scripts.
The `full_sys_data_driven.py` and `att_subsys_data_driven.py` consist of the data-driven approach learning scripts.
The `full_sys_hybrid_parallel_data_driven.py` and `att_subsys_hybrid_parallel_data_driven.py` consist of the hybrid parallel training approach learning scripts and the scripts `full_sys_hybrid_sequential_data_driven.py` and `att_subsys_hybrid_sequential_data_driven.py` consist of the hybrid sequential training approach learning scripts.
To run each of the scripts, we respectivelly run:
```sh
python full_sys_mb_ode.py
python att_subsys_mb_ode.py
python full_sys_sdp.py
python att_subsys_sdp.py
python full_sys_data_driven.py
python att_subsys_data_driven.py
python full_sys_hybrid_parallel_data_driven.py
python att_subsys_hybrid_parallel_data_driven.py
python full_sys_hybrid_sequential_data_driven.py
python att_subsys_hybrid_sequential_data_driven.py
```

## Author
Rodrigo Felizardo


