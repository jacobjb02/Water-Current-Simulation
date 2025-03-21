import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns

def compute_force(velocity, area, drag_coefficient, density):
    speed = np.linalg.norm(velocity)
    return 0.5 * drag_coefficient * density * area * speed**2

def compute_deceleration(velocity, areaBall, CdWater, rhoWater, fraction_in_water, massBall, dt): # CdAir, rhoAir
    #drag_air = compute_force(velocity, areaBall, CdAir, rhoAir)
    drag_water = compute_force(velocity, areaBall, CdWater, rhoWater)
    total_drag = (1 - fraction_in_water) * drag_water + fraction_in_water # drag_air
    deceleration = total_drag / massBall
    speed = np.linalg.norm(velocity)
    if speed != 0:
        return velocity - (velocity / speed) * deceleration * dt
    return velocity

def apply_water_current_force(velocity, areaBall, CdWater, rhoWater, waterSpeed, fraction_in_water, massBall, dt):
    force_magnitude = 0.5 * CdWater * rhoWater * areaBall * abs(waterSpeed) * waterSpeed
    force = np.array([force_magnitude * fraction_in_water, 0, 0])
    acceleration = force / massBall
    return velocity + acceleration * dt

def apply_air_drag(velocity, areaBall, CdAir, rhoAir, massBall, dt):
    air_force_magnitude = 0.5 * CdAir * rhoAir * areaBall * (velocity ** 2)
    air_acceleration = air_force_magnitude / massBall
    speed = np.linalg.norm(velocity)

    if speed != 0:
        return velocity - (velocity / speed) * air_acceleration * dt
    return velocity

def target_angle_from_origin(target_x_pos, target_z_pos):
    target_angle_radians = math.atan2(target_z_pos, target_x_pos)
    target_angle_degrees = math.degrees(target_angle_radians)
    return(target_angle_degrees)

def solution_space_angles_speed(water_speed, target_x_pos, target_z_pos, target_width):
    # Constants
    CdAir = 0.47
    CdWater = 0.47
    rhoAir = 1.225
    rhoWater = 1000
    massBall = 1.0
    radiusBall = 0.0457
    areaBall = np.pi * radiusBall ** 2

    #Air Drag Application Circle
    air_drag_circle_pos = [0,0,0]
    air_drag_circle_radius = 0.1 # first 10 cm of launch
    air_drag_circle_pos_x = 0
    air_drag_circle_pos_z = 0

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
    num_angles = 181
    angle_range = np.linspace(0, 180, num_angles)
    num_speeds = 51
    speed_range = np.linspace(0, 5, num_speeds)

    # Data Storage
    count_hits = 0
    angles, hit_angles, miss_angles = [], [], []
    speeds, hit_speeds, miss_speeds = [], [], []
    min_distance_list = []
    final_x_list, final_z_list = [], []

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
                # air drag for first 0.1 m on z axis
                direction_air = position - air_drag_circle_pos
                distance_to_air = np.linalg.norm(direction_air[[0, 2]])

                if distance_to_air < air_drag_circle_radius:
                    velocity = apply_air_drag(velocity, areaBall, CdAir, rhoAir, massBall, dt)
                else:
                    frac_water = 0.5
                    velocity = compute_deceleration(velocity, areaBall, CdWater, rhoWater, frac_water,
                                                massBall, dt) # CdAir, rhoAir
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

    # Air Drag Application region
    for ax in axes:
        air_drag_circle = plt.Circle((air_drag_circle_pos_x, air_drag_circle_pos_z), air_drag_circle_radius, color='black', alpha=0.25,
                                   edgecolor='black', zorder=10)
        ax.add_patch(air_drag_circle)

    ax.set_xlim([-3, 3])
    ax.set_ylim([0, 3])
    plt.show()

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
    plt.show()

    # Successful angle range
    tmp_angle = tuple(hit_angles)

    # Compute minimum and maximum hit angles
    min_angle = min(tmp_angle)
    max_angle = max(tmp_angle)

    # Compute median hit angle
    median_angle = np.median(tmp_angle)

    # Hit Angle Span
    hit_span_angle = max_angle - min_angle

    # Successful speed range
    tmp_speed = tuple(hit_speeds)

    # Compute minimum and maximum hit angles
    min_speed = min(tmp_speed)
    max_speed = max(tmp_speed)

    # Compute median hit angle
    median_speed = np.median(tmp_speed)

    # Hit Speed Span
    hit_span_speed = max_speed - min_speed

    print(f"Min Hit Angle: {min_angle:.1f} | Max Hit Angle: {max_angle:.1f} | Median Hit Angle: {median_angle:.1f}")
    print(f"Hit Angle Range: {hit_span_angle:.1f}")
    print(f"Target Angle: {target_angle:.1f}")
    print("========================")
    print(f"Min Hit Speed: {min_speed:.1f} | Max Hit Speed: {max_speed:.1f} | Median Hit Speed: {median_speed:.1f}")
    print(f"Hit Speed Range: {hit_span_speed:.1f}")

    # Save data
    df = pd.DataFrame({
        "Launch Angle (°)": angles,
        "Launch Speed (m/s)": speeds,
        "Min Distance to Target": min_distance_list,
        "Final x coordinate": final_x_list,
        "Final z coordinate": final_z_list,
    })
    df.to_csv('launch_data.csv', index=False)

    # Heatmap
    heatmap_data = df.pivot(index="Launch Speed (m/s)", columns="Launch Angle (°)", values="Min Distance to Target")

    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(heatmap_data, cmap="RdBu_r", annot=False, linewidths=0.5)

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
    plt.show()

# 9 targets

# Water Speed = 0 m/s
# ===============================================
    # centered target
#solution_space_angles_speed(0, 0, 1.5, 0.2)
    # left target
#solution_space_angles_speed(0, -0.7, 1.5, 0.2)
    # right target
#solution_space_angles_speed(0, 0.7, 1.5, 0.2)


# 4 targets:
# Water Speed = -2 m/s
# ===============================================
solution_space_angles_speed(0, 0, 1.5, 0.2) # check: 60.7 degrees solution space
solution_space_angles_speed(-2, 0, 1.5, 0.2) # check: 60.7 degrees solution space
#solution_space_angles_speed(-2, -0.25,1.5, 0.4) # check: 60.7 degrees solution space

#solution_space_angles_speed(-2, 0, 1.25, 0.2) # check: 60.7 degrees solution space
#solution_space_angles_speed(-2, -0.25, 1.25, 0.2) # check: 60.7 degree solution space

#solution_space_angles_speed(-2, -0.3, 1.85, 0.2) # check: 60.7 degree solution space

# Same water speed but significantly different target location (x axis):
#solution_space_angles_speed(-2, -0.75, 1.5, 0.2) # check: 89 degree solution space

# It seems we have to increase the target_z position in proportion to decreasing the target_x position (same direction as water current)
# in order to have a consistent 60 degree range of hit_angles










# left hand
#solution_space_angles_speed(4, 1, 1, 0.2)

# threshold with controller velocity, reduce number of failed throws by indicating if they didnnt throw faster and re-use trial
# recycle trials, one trial number, then success speed trial nums; if speed was not reached, ball sinks
# ensure angle solution space doesnt exceed ~60 degrees
# 9 different target locations + two extra with generalization
# png images store
# end collider on x meter for solution space


# slope -> beach env -> water current applied immediately
# reltuve to target angle
# dashedline for target angle

# make 0 min distance a distinct colour

# allow us to manipulate target x and z in json

# gradual transition of fraction_in_water, esp for transition from air_drag -> water_drag

# jupitry notebook

# ask for URPP ID!
# TYPE E proposal