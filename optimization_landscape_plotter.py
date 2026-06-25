import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_landscape_plot():
    print("Calculating optimization landscape...")
    
    # 1. Define the Robot's Cartesian Workspace
    x = np.linspace(-0.5, 1.5, 400)
    y = np.linspace(-0.5, 1.5, 400)
    X, Y = np.meshgrid(x, y)

    # 2. Set Parameters matching controller.py
    target_x, target_y = 1.2, 1.2
    obs_x, obs_y = 0.5, 0.5
    r_safe = 0.3
    W_obs = 50000.0  # Scaled slightly for visual plotting

    # 3. Calculate Target Tracking Cost (The Convex Bowl)
    # J_track = 500 * ||p_ee - p_target||^2
    J_track = 500.0 * ((X - target_x)**2 + (Y - target_y)**2)

    # 4. Calculate Obstacle Slack Penalty (The Non-Convex Pillar)
    dist = np.sqrt((X - obs_x)**2 + (Y - obs_y)**2)
    # Slack is required if distance is strictly less than r_safe
    slack = np.maximum(0, r_safe - dist)
    J_obs = W_obs * slack
    

    # 5. Combine for Total Cost
    Z = J_track + J_obs
    
    # Clip the Z values strictly for visual scaling, otherwise the 
    # massive 50,000 slack penalty completely visually dwarfs the tracking bowl
    Z_clipped = np.clip(Z, 0, 4000)

    # --- Generate the 3D Academic Plot ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z_clipped, cmap='plasma', edgecolor='none', alpha=0.9)

    # Add 2D Contour projection on the floor of the plot
    ax.contour(X, Y, Z_clipped, zdir='z', offset=-500, cmap='plasma', levels=30)

    # Annotations & Markers
    ax.scatter(target_x, target_y, 0, color='green', s=100, label='Target (Global Minimum)', zorder=5)
    ax.scatter(obs_x, obs_y, 4000, color='red', s=100, label='Obstacle Center', zorder=5)

    # Labels and Titles
    ax.set_title('optimization cost function', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Workspace X (meters)', fontsize=12, labelpad=10)
    ax.set_ylabel('Workspace Y (meters)', fontsize=12, labelpad=10)
    ax.set_zlabel('Total Cost Objective (J)', fontsize=12, labelpad=10)
    ax.set_zlim(-500, 4000)

    # Adjust viewing angle for the best perspective of the "split" valley
    ax.view_init(elev=35, azim=-125)
    
    ax.legend(loc='upper right', fontsize=12)
    plt.tight_layout()

    # Save to file
    plt.savefig('optimization_landscape.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'optimization_landscape.png'!")
    plt.show()

if __name__ == '__main__':
    generate_landscape_plot()