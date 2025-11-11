import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml

# -------------------------
# Utility functions
# -------------------------
def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def line_sphere_collision(p1, p2, sphere_center, sphere_r):
    """Check if segment p1-p2 intersects sphere"""
    p1 = np.array(p1); p2 = np.array(p2); c = np.array(sphere_center)
    d = p2 - p1
    if np.allclose(d, 0):
        return np.linalg.norm(p1 - c) <= sphere_r
    t = np.dot(c - p1, d) / np.dot(d, d)
    t = max(0.0, min(1.0, t))
    closest = p1 + t * d
    return np.linalg.norm(closest - c) <= (sphere_r + 1e-9)

def segment_collision_check(p1, p2, obstacles):
    for (c, r) in obstacles:
        if line_sphere_collision(p1, p2, c, r):
            return True
    return False

def path_length(path):
    if path is None or len(path) < 2: return 0.0
    return sum(dist(path[i], path[i+1]) for i in range(len(path)-1))

# -------------------------
# Sampling
# -------------------------
def sample_free(bounds, goal=None, goal_bias=0.0):
    if goal is not None and random.random() < goal_bias:
        return tuple(goal)
    return (random.uniform(*bounds[0]), random.uniform(*bounds[1]), random.uniform(*bounds[2]))

# -------------------------
# Basic RRT
# -------------------------
def rrt_basic(start, goal, obstacles, bounds, max_iters=5000, step_size=20.0, 
              goal_radius=10.0, goal_bias=0.05):
    nodes = [start]
    parents = {0: None}
    start_time = time.time()
    
    for i in range(max_iters):
        x_rand = sample_free(bounds, goal=goal, goal_bias=goal_bias)
        dists = [dist(n, x_rand) for n in nodes]
        idx = int(np.argmin(dists))
        x_near = nodes[idx]
        
        v = np.array(x_rand) - np.array(x_near)
        L = np.linalg.norm(v)
        if L == 0:
            continue
        step = (v / L) * min(step_size, L)
        x_new = tuple(np.array(x_near) + step)
        
        if not segment_collision_check(x_near, x_new, obstacles):
            nodes.append(x_new)
            parents[len(nodes)-1] = idx
            if dist(x_new, goal) <= goal_radius:
                path = [goal]
                cur = len(nodes)-1
                while cur is not None:
                    path.append(nodes[cur])
                    cur = parents[cur]
                path.reverse()
                runtime = time.time() - start_time
                return path, nodes, parents, runtime
    
    runtime = time.time() - start_time
    return None, nodes, parents, runtime

# -------------------------
# APF utilities
# -------------------------
def compute_apf_force(x_near, goal, obstacles, K_att=1.0, K_rep=0.3, d0=60.0):
    """Compute APF force at position x_near"""
    v_att = np.array(goal) - np.array(x_near)
    d_att = np.linalg.norm(v_att)
    F_att = (K_att * (v_att / (d_att + 1e-9))) if d_att > 0 else np.zeros(3)

    F_rep = np.zeros(3)
    for (c, r) in obstacles:
        v_obs = np.array(x_near) - np.array(c)
        dist_center = np.linalg.norm(v_obs)
        d_obs = dist_center - r
        
        if d_obs <= 0:
            if dist_center > 1e-6:
                F_rep += K_rep * (v_obs / (dist_center + 1e-9)) * (1.0 / (1e-3))
            else:
                F_rep += K_rep * np.random.randn(3)
        elif d_obs <= d0:
            mag = K_rep * (1.0 / (d_obs ** 2)) * (1.0/d_obs - 1.0/d0)
            if dist_center > 1e-9:
                F_rep += mag * (v_obs / dist_center)
    
    F_total = F_att + F_rep
    return F_total, F_att, F_rep

# -------------------------
# APF-guided RRT
# -------------------------
def rrt_apf_guided(start, goal, obstacles, bounds, max_iters=5000, r_step=25.0, 
                   goal_radius=10.0, goal_bias=0.07, K_att=1.0, K_rep=0.3, d0=80.0):
    nodes = [start]
    parents = {0: None}
    start_time = time.time()
    
    for it in range(max_iters):
        x_rand = sample_free(bounds, goal=goal, goal_bias=goal_bias)
        dists = [dist(n, x_rand) for n in nodes]
        idx = int(np.argmin(dists))
        x_near = nodes[idx]
        
        v_rand = np.array(x_rand) - np.array(x_near)
        norm_v_rand = np.linalg.norm(v_rand)
        v_rand_unit = (v_rand / norm_v_rand) if norm_v_rand>0 else np.zeros(3)
        
        F_total, F_att, F_rep = compute_apf_force(x_near, goal, obstacles, 
                                                   K_att=K_att, K_rep=K_rep, d0=d0)
        
        v_goal = np.array(goal) - np.array(x_near)
        nv_goal = v_goal / (np.linalg.norm(v_goal)+1e-9)
        nv_obs = -F_rep / (np.linalg.norm(F_rep)+1e-9) if np.linalg.norm(F_rep)>1e-9 else np.zeros(3)
        
        r1, r2, r3 = 0.6, 0.3, 0.6
        combined = r1 * v_rand_unit + r2 * nv_goal + r3 * nv_obs
        if np.linalg.norm(combined) == 0:
            combined = nv_goal
        
        rep_mag = np.linalg.norm(F_rep)
        step_adj = r_step * (0.2 + 0.5 * random.random())
        if rep_mag > 0.5:
            step_adj *= 0.5
        
        direction = (combined / np.linalg.norm(combined)) * step_adj
        x_new = tuple(np.array(x_near) + direction)
        
        if not segment_collision_check(x_near, x_new, obstacles):
            nodes.append(x_new)
            parents[len(nodes)-1] = idx
            if dist(x_new, goal) <= goal_radius:
                path = [goal]
                cur = len(nodes)-1
                while cur is not None:
                    path.append(nodes[cur])
                    cur = parents[cur]
                path.reverse()
                runtime = time.time() - start_time
                return path, nodes, parents, runtime
    
    runtime = time.time() - start_time
    return None, nodes, parents, runtime

# -------------------------
# Path pruning
# -------------------------
def prune_path(path, obstacles):
    if path is None:
        return None
    pruned = [path[0]]
    i = 0
    while i < len(path)-1:
        j = len(path)-1
        while j > i+1:
            if not segment_collision_check(path[i], path[j], obstacles):
                break
            j -= 1
        pruned.append(path[j])
        i = j
    return pruned

# -------------------------
# Environment
# -------------------------
def create_random_spheres(num=6, bounds=((0,500),(0,500),(0,300)), rmin=20, rmax=60, seed=None):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    obstacles = []
    for _ in range(num):
        x = random.uniform(bounds[0][0]+20, bounds[0][1]-20)
        y = random.uniform(bounds[1][0]+20, bounds[1][1]-20)
        z = random.uniform(bounds[2][0]+20, bounds[2][1]-20)
        r = random.uniform(rmin, rmax)
        obstacles.append(((x,y,z), r))
    return obstacles

# -------------------------
# ROS EXPORT FUNCTIONS (NEW)
# -------------------------
def export_to_ros_yaml(path, obstacles, filename='path_for_ros.yaml'):
    """Export path and obstacles in ROS-compatible YAML format"""
    if path is None:
        print(f"Cannot export None path to {filename}")
        return
    
    data = {
        'path': {
            'waypoints': [
                {'position': {'x': float(p[0]/1000.0), 'y': float(p[1]/1000.0), 'z': float(p[2]/1000.0)}}
                for p in path
            ],
            'frame_id': 'world'
        },
        'obstacles': [
            {
                'type': 'sphere',
                'center': {'x': float(c[0]/1000.0), 'y': float(c[1]/1000.0), 'z': float(c[2]/1000.0)},
                'radius': float(r/1000.0)
            }
            for (c, r) in obstacles
        ]
    }
    
    with open(filename, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"Exported ROS-compatible data to {filename}")

def export_obstacles_csv(obstacles, filename='obstacles.csv'):
    """Export obstacles for easy loading"""
    obstacle_data = []
    for (center, radius) in obstacles:
        obstacle_data.append([center[0], center[1], center[2], radius])
    np.savetxt(filename, obstacle_data, delimiter=',', 
               header='x_mm,y_mm,z_mm,radius_mm', comments='')
    print(f"Saved obstacles to {filename}")

def create_rviz_marker_file(obstacles, filename='obstacles.marker'):
    """Create RViz marker config for visualization"""
    markers = []
    for idx, (center, radius) in enumerate(obstacles):
        marker = {
            'id': idx,
            'type': 'sphere',
            'pose': {
                'position': {'x': center[0]/1000.0, 'y': center[1]/1000.0, 'z': center[2]/1000.0},
                'orientation': {'w': 1.0, 'x': 0.0, 'y': 0.0, 'z': 0.0}
            },
            'scale': {'x': radius/500.0, 'y': radius/500.0, 'z': radius/500.0},
            'color': {'r': 0.5, 'g': 0.5, 'b': 0.5, 'a': 0.7}
        }
        markers.append(marker)
    
    with open(filename, 'w') as f:
        yaml.dump({'markers': markers}, f)
    print(f"Saved RViz markers to {filename}")

# -------------------------
# Main experiment (ENHANCED)
# -------------------------
def run_experiment(seed=1, show_plot=True, export_ros=True):
    random.seed(seed); np.random.seed(seed)
    bounds = ((0,500),(0,500),(0,300))
    start = (30.0, 30.0, 40.0)
    goal  = (450.0, 450.0, 240.0)
    obstacles = create_random_spheres(num=7, bounds=bounds, rmin=25, rmax=70, seed=seed)
    
    def inside_any(p, obs):
        for (c,r) in obs:
            if dist(p, c) <= r:
                return True
        return False
    
    tries = 0
    while (inside_any(start, obstacles) or inside_any(goal, obstacles)) and tries < 10:
        obstacles = create_random_spheres(num=7, bounds=bounds, rmin=25, rmax=70, seed=seed+tries+1)
        tries += 1

    print("Running baseline RRT...")
    path_b, nodes_b, parents_b, t_b = rrt_basic(start, goal, obstacles, bounds,
                                                max_iters=8000, step_size=30.0, 
                                                goal_radius=20.0, goal_bias=0.05)
    if path_b is None:
        print("Baseline RRT: FAILED")
    else:
        pruned_b = prune_path(path_b, obstacles)
        print(f"Baseline: {len(nodes_b)} nodes, {path_length(path_b):.2f} mm")

    print("Running APF-guided RRT...")
    path_i, nodes_i, parents_i, t_i = rrt_apf_guided(start, goal, obstacles, bounds,
                                                     max_iters=8000, r_step=35.0, 
                                                     goal_radius=20.0, goal_bias=0.07,
                                                     K_att=1.0, K_rep=0.3, d0=100.0)
    if path_i is None:
        print("APF-guided RRT: FAILED")
    else:
        pruned_i = prune_path(path_i, obstacles)
        print(f"Improved: {len(nodes_i)} nodes, {path_length(path_i):.2f} mm")

    # Print metrics
    print("\n--- Metrics ---")
    if path_b is not None:
        print(f"Baseline: {t_b:.3f}s, {len(nodes_b)} nodes, {path_length(pruned_b):.2f} mm")
    if path_i is not None:
        print(f"Improved: {t_i:.3f}s, {len(nodes_i)} nodes, {path_length(pruned_i):.2f} mm")

    # Save CSV outputs
    out1 = "path_points_baseline.csv"
    out2 = "path_points_improved.csv"
    if path_b is not None:
        np.savetxt(out1, np.array(path_b), delimiter=",", header="x,y,z", comments='')
        print(f"Saved baseline to {out1}")
    if path_i is not None:
        np.savetxt(out2, np.array(path_i), delimiter=",", header="x,y,z", comments='')
        print(f"Saved improved to {out2}")

    # ROS EXPORT (NEW)
    if export_ros:
        print("\n--- Exporting ROS-compatible files ---")
        if path_i is not None:
            export_to_ros_yaml(path_i, obstacles, 'path_improved_ros.yaml')
            export_to_ros_yaml(pruned_i, obstacles, 'path_improved_pruned_ros.yaml')
        if path_b is not None:
            export_to_ros_yaml(path_b, obstacles, 'path_baseline_ros.yaml')
        
        export_obstacles_csv(obstacles)
        create_rviz_marker_file(obstacles)

    # Visualization
    if show_plot:
        fig = plt.figure(figsize=(15,7))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

        # Draw obstacles
        for (c,r) in obstacles:
            u = np.linspace(0, 2*np.pi, 24)
            v = np.linspace(0, np.pi, 12)
            x = c[0] + r * np.outer(np.cos(u), np.sin(v))
            y = c[1] + r * np.outer(np.sin(u), np.sin(v))
            z = c[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
            ax1.plot_surface(x, y, z, color='gray', alpha=0.25, linewidth=0)
            ax2.plot_surface(x, y, z, color='gray', alpha=0.25, linewidth=0)

        # Baseline plot
        if path_b is not None:
            pb = np.array(path_b)
            ax1.plot(pb[:,0], pb[:,1], pb[:,2], '-r', linewidth=2, label='Path')
            ax1.scatter(pb[:,0], pb[:,1], pb[:,2], c='r', s=20)
        nodes_b_arr = np.array(nodes_b)
        ax1.scatter(nodes_b_arr[:,0], nodes_b_arr[:,1], nodes_b_arr[:,2], 
                   s=6, alpha=0.15, c='blue', label='Nodes')
        ax1.scatter([start[0]], [start[1]], [start[2]], c='green', s=100, 
                   marker='o', label='Start', edgecolors='black', linewidths=2)
        ax1.scatter([goal[0]], [goal[1]], [goal[2]], c='purple', s=100, 
                   marker='*', label='Goal', edgecolors='black', linewidths=2)
        ax1.set_title(f'Baseline RRT\n({len(nodes_b)} nodes, {t_b:.2f}s)')
        ax1.set_xlabel('X (mm)'); ax1.set_ylabel('Y (mm)'); ax1.set_zlabel('Z (mm)')
        ax1.set_xlim(bounds[0]); ax1.set_ylim(bounds[1]); ax1.set_zlim(bounds[2])
        ax1.legend()

        # Improved plot
        if path_i is not None:
            pi = np.array(path_i)
            ax2.plot(pi[:,0], pi[:,1], pi[:,2], '-b', linewidth=2, label='Path')
            ax2.scatter(pi[:,0], pi[:,1], pi[:,2], c='b', s=20)
        nodes_i_arr = np.array(nodes_i)
        ax2.scatter(nodes_i_arr[:,0], nodes_i_arr[:,1], nodes_i_arr[:,2], 
                   s=6, alpha=0.15, c='blue', label='Nodes')
        ax2.scatter([start[0]], [start[1]], [start[2]], c='green', s=100, 
                   marker='o', label='Start', edgecolors='black', linewidths=2)
        ax2.scatter([goal[0]], [goal[1]], [goal[2]], c='purple', s=100, 
                   marker='*', label='Goal', edgecolors='black', linewidths=2)
        ax2.set_title(f'APF-guided RRT\n({len(nodes_i)} nodes, {t_i:.2f}s)')
        ax2.set_xlabel('X (mm)'); ax2.set_ylabel('Y (mm)'); ax2.set_zlabel('Z (mm)')
        ax2.set_xlim(bounds[0]); ax2.set_ylim(bounds[1]); ax2.set_zlim(bounds[2])
        ax2.legend()

        plt.tight_layout()
        plt.savefig('apf_rrt_comparison.png', dpi=150, bbox_inches='tight')
        print("Saved plot to apf_rrt_comparison.png")
        plt.show()

    return {
        "baseline": {"path": path_b, "nodes": nodes_b, "runtime": t_b},
        "improved": {"path": path_i, "nodes": nodes_i, "runtime": t_i},
        "obstacles": obstacles
    }

# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("APF-guided RRT Path Planning")
    print("Enhanced version with ROS export")
    print("=" * 60)
    results = run_experiment(seed=2, show_plot=True, export_ros=True)
    print("\n✓ Done! Check the generated files:")
    print("  - CSV files: path_points_*.csv, obstacles.csv")
    print("  - ROS files: path_*_ros.yaml, obstacles.marker")
    print("  - Plot: apf_rrt_comparison.png")
