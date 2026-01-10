import numpy as np
import sys
import os

def analyze_trajectory(npz_path):
    print(f"Analyzing {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)
    
    if "traj_0" not in data:
        print("Error: Expected receding horizon format (traj_0)")
        return

    traj = data["traj_0"]  # (T, 7)
    goal = data["goal_0"]
    dt = 0.02
    
    # Calculate velocities
    vel = np.diff(traj, axis=0) / dt
    speed = np.linalg.norm(vel, axis=1)
    
    # Calculate distance to goal in Joint Space
    dist_to_goal = np.linalg.norm(traj - goal, axis=1)
    
    print(f"Trajectory length: {len(traj)} steps")
    print(f"Final distance to goal: {dist_to_goal[-1]:.4f} rad")
    
    # Check for "stuck" condition
    # Stuck if speed is low but distance to goal is high
    stuck_threshold_speed = 0.05
    stuck_threshold_dist = 0.5
    
    # Look at the last 50 steps
    if len(speed) > 50:
        avg_final_speed = np.mean(speed[-50:])
        print(f"Average speed (last 50 steps): {avg_final_speed:.4f} rad/s")
        
        if avg_final_speed < stuck_threshold_speed and dist_to_goal[-1] > stuck_threshold_dist:
            print("STATUS: STUCK")
            print("The robot has stopped moving significantly but is far from the goal.")
        else:
            print("STATUS: MOVING or REACHED")
    
    # Print progress
    print("\nStep-wise analysis (every 50 steps):")
    for i in range(0, len(traj), 50):
        s = 0 if i==0 else speed[i-1]
        d = dist_to_goal[i]
        print(f"  Step {i:03d}: Speed={s:.4f}, DistToGoal={d:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_stuck.py <npz_file>")
        sys.exit(1)
    
    analyze_trajectory(sys.argv[1])
