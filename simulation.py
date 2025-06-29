import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns
import time
import itertools
import os
import multiprocessing
from tqdm import tqdm

# prevents if functions are ran in different script
if __name__ != '__main__':
    multiprocessing.freeze_support()

def compute_force(velocity, area, drag_coefficient, density):
    speed = np.linalg.norm(velocity)
    return 0.5 * drag_coefficient * density * area * speed**2

def compute_deceleration(velocity, areaBall, CdAir, rhoAir, CdWater, rhoWater, fraction_in_water, massBall, dt):
    speed = np.linalg.norm(velocity)
    if speed == 0:
        return velocity

    # drag
    # drag_air = compute_force(speed, areaBall, CdAir, rhoAir)
    drag_water = compute_force(speed, areaBall, CdWater, rhoWater)

    # Weighted average drag based on fraction in water (Unity uses f = 0.5)
    total_drag = fraction_in_water * drag_water

    # Convert to scalar deceleration (a = F/m)
    deceleration = total_drag / massBall

    # Reduce speed
    new_speed = max(speed - deceleration * dt, 0)

    # Apply it in the same direction
    return (velocity / speed) * new_speed


def apply_water_current_force(velocity, areaBall, CdWater, rhoWater, waterSpeed, fraction_in_water, massBall, dt):
    drag_force = compute_force(abs(waterSpeed), areaBall, CdWater, rhoWater)

    total_force = 0.5 * fraction_in_water * drag_force

    # Convert to acceleration and adjust velocity in X-direction
    delta_v = (total_force / massBall) * dt
    velocity[0] -= delta_v  # water flows leftward in Unity

    return velocity


def target_angle_from_origin(target_x_pos, target_z_pos):
    target_angle_radians = math.atan2(target_z_pos, target_x_pos)
    target_angle_degrees = math.degrees(target_angle_radians)
    return(target_angle_degrees)

def solution_space_angles_speed(water_speed, target_x_pos, target_z_pos, target_width):

    # Measure time since running function
    start = time.time()

    # Define folder structure
    data_folder = f"simulation_data/launch_data_{water_speed}"
    figures_folder = os.path.join(data_folder, "simulation_figures")

    # Create folders if they don't exist
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(figures_folder, exist_ok=True)

    # Constants
    CdAir = 0
    CdWater = 0.47
    rhoAir = 0
    rhoWater = 1000
    massBall = 1.0
    radiusBall = 0.0457
    areaBall = np.pi * radiusBall ** 2
    frac_water = 0.5

    # Target Area
    target_y_pos = 0  # Keeping Y for later
    target_radius = (target_width / 2)
    target_pos = np.array([target_x_pos, target_y_pos, target_z_pos])

    # Target Angle (in degrees) relative to origin
    target_angle = target_angle_from_origin(target_x_pos, target_z_pos)
    # print(target_angle)

    # Simulation
    dt = 0.02
    time_total = 5

    # Launch Parameters
    num_angles = 115
    angle_range = np.linspace(15, 130, num_angles)
    num_speeds =  50
    speed_range = np.linspace(0.5, 5.5, num_speeds)

    # Data Storage
    count_hits = 0
    angles, hit_angles, miss_angles = [], [], []
    speeds, hit_speeds, miss_speeds = [], [], []
    min_distance_list = []
    final_x_list, final_z_list = [], []
    target_hit_list = []

    # Colormap
    cmap = cm.viridis
    norm = mcolors.Normalize(vmin=min(speed_range), vmax=max(speed_range))

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    fig.subplots_adjust(right=0.85)
    axes[0].set_title(f'Trajectories Hitting Target: {water_speed} (m/s)')
    axes[1].set_title(f'Trajectories Missing the Target: {water_speed} (m/s)')

    for ax in axes:
        ax.set_xlabel('X Position')
        ax.set_ylabel('Z Position')
        ax.grid(True)


    # Loop over angles
    for angle_deg in angle_range:
        for launch_speed in speed_range:
            angle_rad = np.radians(angle_deg)

            # Velocity is now (X, Y, Z) but Y is ignored in calculations
            velocity = np.array([
                launch_speed * np.cos(angle_rad),  # X
                0,  # Y (Ignored for now)
                launch_speed * np.sin(angle_rad)  # Z
            ])

            # Position is (X, Y, Z)
            position = np.array([0.0, 0.0, 0.0])
            positions = [position.copy()]

            hits_target = False
            closest_distance = float('inf')

            for t in np.arange(0, time_total, dt):
                velocity = compute_deceleration(velocity, areaBall, CdAir, rhoAir, CdWater, rhoWater, frac_water,
                                                massBall, dt)
                velocity = apply_water_current_force(velocity, areaBall, CdWater, rhoWater, water_speed, frac_water,
                                                     massBall, dt)

                position += velocity * dt
                positions.append(position.copy())

                # Closest Point Calculation to target (X, Y, Z) but Y is ignored
                direction = position - target_pos
                distance_to_center = np.linalg.norm(direction[[0, 2]])  # Only (X, Z)

                if distance_to_center <= target_radius:
                    closest_point = position.copy()  # Ball is inside or on target center
                else:
                    closest_point = target_pos.copy().astype(float)
                    closest_point[[0, 2]] += (direction[[0, 2]] / distance_to_center) * target_radius

                distance_to_surface = max(0, distance_to_center - target_radius)

                if distance_to_surface < closest_distance:
                    closest_distance = distance_to_surface

                if distance_to_center <= target_radius:
                    hits_target = True
                    break # stop simulating trajectory if target is hit

                if position[2] < 0 or abs(position[0]) >= 3 or position[2] > 2.56:
                    break

            # Convert positions list to np array for each trajectory
            positions = np.array(positions)

            # Append final position for each trajectory
            final_x_list.append(positions[-1, 0])  # Final x pos for trajectory
            final_z_list.append(positions[-1, 2])  # Final z pos for trajectory

            # Store trajectory parameters
            angles.append(angle_deg)
            speeds.append(launch_speed)
            min_distance_list.append(closest_distance)
            target_hit_list.append(hits_target)

            if hits_target:
                count_hits += 1
                hit_angles.append(angle_deg)
                hit_speeds.append(launch_speed)
            else:
                miss_angles.append(angle_deg)
                miss_speeds.append(launch_speed)

            positions = np.array(positions)
            ax = axes[0] if hits_target else axes[1]

            # Plot trajectory
            ax.plot(positions[:, 0], positions[:, 2], color=cmap(norm(launch_speed)), alpha=0.7)

    # Colorbar
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='Launch Speed (m/s)')

    # Target visualization
    for ax in axes:
        target_circle = plt.Circle((target_x_pos, target_z_pos), target_radius, color='red', alpha=0.25,
                                   edgecolor='black', zorder=10)
        ax.add_patch(target_circle)

    ax.set_xlim([-3, 3])
    ax.set_ylim([0, 3])
    #plt.show()

    plt.figure(figsize=(10, 5))
    ax = plt.axes()
    ax.set_facecolor('black')

    plt.scatter(hit_angles, hit_speeds, color='green', label='Hit', alpha=0.8, edgecolors="black")
    plt.scatter(miss_angles, miss_speeds, color='red', label='Miss', alpha=0.4, edgecolors="black")

    plt.xlabel("Launch Angle (°)")
    plt.ylabel("Launch Speed (m/s)")
    plt.title("Launch Angle vs. Launch Speed")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
   # plt.show()

    # Successful angle range
    if len(hit_angles) > 0:
        tmp_angle = tuple(hit_angles)

        # Compute minimum and maximum hit angles
        min_angle = min(tmp_angle)
        max_angle = max(tmp_angle)

        # Compute median hit angle
        median_angle = np.median(tmp_angle)

        # Hit Angle Span
        hit_span_angle = max_angle - min_angle

        print(f"Min Hit Angle: {min_angle:.1f} | Max Hit Angle: {max_angle:.1f} | Median Hit Angle: {median_angle:.1f}")
    else:
        print("No hit angles recorded.")

    # Successful speed range
    if len(hit_speeds) > 0:
        tmp_speed = tuple(hit_speeds)

        # Compute minimum and maximum hit angles
        min_speed = min(tmp_speed)
        max_speed = max(tmp_speed)

        # Compute median hit angle
        median_speed = np.median(tmp_speed)

        # Hit Speed Span
        hit_span_speed = max_speed - min_speed

        print(f"Min Hit Speed: {min_speed:.1f} | Max Hit Speed: {max_speed:.1f} | Median Hit Speed: {median_speed:.1f}")
    else:
        print("No hit speeds recorded.")

    # Save data
    df = pd.DataFrame({
        "launch_angle_deg": angles,
        "launch_speed_m_s": speeds,
        "target_angle": target_angle,
        "target_x_pos": target_x_pos,
        "target_z_pos": target_z_pos,
        "target_radius": target_radius,
        "min_distance_to_target": min_distance_list,
        "target_hit": target_hit_list,
        "final_x_coordinate": final_x_list,
        "final_z_coordinate": final_z_list,
    })
    filename = os.path.join(data_folder, f'simulation_data_{water_speed}.csv')
    if os.path.exists(filename):
        # Append to the file without writing the header again
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        # File doesn't exist yet, so create it with the header
        df.to_csv(filename, mode='w', header=True, index=False)

    # Heatmap
    heatmap_data = df.pivot(index="launch_speed_m_s", columns="launch_angle_deg", values="min_distance_to_target")

    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(heatmap_data, cmap="RdBu_r", annot=False, linewidths=0.5)

    angle_columns = heatmap_data.columns.values

    # Find closest angle
    target_col_idx = np.argmin(np.abs(angle_columns - target_angle))

    # Convert target radius to angular measure (radians)
    theta_radians = math.atan(target_radius / target_z_pos)
    # Convert the radians to degrees
    theta_degrees = math.degrees(theta_radians)

    target_angle_neg = target_angle - theta_degrees
    target_angle_pos = target_angle + theta_degrees

    target_col_idx_neg = np.argmin(np.abs(angle_columns - target_angle_neg))
    target_col_idx_pos = np.argmin(np.abs(angle_columns - target_angle_pos))

    # target angle vertical lines
    ax.axvline(x=target_col_idx, color='red', linestyle='--', linewidth=3)
    ax.axvline(x=target_col_idx_neg, color='red', linestyle='--', linewidth=2)
    ax.axvline(x=target_col_idx_pos, color='red', linestyle='--', linewidth=2)

    # Flip the y-axis to have lowest values at bottom
    ax.invert_yaxis()

    # Round and clean up Y-axis tick labels
    y_labels = [f"{float(label.get_text()):.1f}" for label in ax.get_yticklabels()]
    ax.set_yticklabels(y_labels, rotation=0)

    # Clean up X-axis tick labels
    x_labels = [f"{float(label.get_text()):.1f}" for label in ax.get_xticklabels()]
    ax.set_xticklabels(x_labels, rotation=45, ha='right')

    plt.title(f"{water_speed} (m/s) Water Speed | Target Angle: {target_angle:.1f} (°) | Target Position: (x={target_x_pos}, z={target_z_pos}) | Target Radius = {target_radius} (m)", fontsize=14)
    plt.xlabel("Launch Angle (°)", fontsize=12)
    plt.ylabel("Launch Speed (m/s)", fontsize=12)

    plt.tight_layout()


    #plt.show()
    os.makedirs('plots', exist_ok=True)
    figure_filename = os.path.join(figures_folder,
                                   f'heatmap_{water_speed}_{target_x_pos}_{target_z_pos}_{target_radius}.png')
    # replace figures
    if os.path.exists(figure_filename):
        try:
            os.remove(figure_filename)
        except PermissionError:
            # give the OS a moment to release the handle
            time.sleep(0.01)
            os.remove(figure_filename)

    # save new figure
    plt.savefig(figure_filename, dpi=300, bbox_inches='tight')

    # close figure
    plt.close(fig)

    #print(f"Target Angle: {target_angle:.1f}")
    print("========================")
    print(f"Saved CSV: {filename}")
    print("========================")
    print(f"Saved Heatmap: {figure_filename}")

def run_simulation(params):
    water_speed, target_x, target_z, target_width = params
    solution_space_angles_speed(water_speed, target_x, target_z, target_width)
    return params  # or nothing

if __name__ == '__main__':
    # Measure time since running function
    start = time.time()

    # Define the parameter ranges
    water_speed_values = [-2.5, -3.0]
    target_x_values = [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    target_z_values = [1.4, 1.3]
    target_width_values = [0.15]

    # Create a list of all parameter combinations
    params_list = list(itertools.product(water_speed_values, target_x_values, target_z_values, target_width_values))

    # Calculate 30% of available cores
    total_cores = multiprocessing.cpu_count()
    num_workers = max(1, int(total_cores * 0.30))
    print(f"Total cores: {total_cores}, Using: {num_workers} cores for processing")

    # Run simulations in parallel
    with multiprocessing.Pool(processes=num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(run_simulation, params_list), total=len(params_list)):
            pass


        # end time
        end = time.time()
        end_time = end - start

    print("========================")
    print("All simulations completed!")
    print("========================")
    print(f"Function took {end_time/60:.2f} minutes to complete")
    print("========================")
