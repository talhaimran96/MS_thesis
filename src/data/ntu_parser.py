import os
import numpy as np

def read_skeleton_file(file_path: str, max_bodies: int = 2, num_joints: int = 25):
    """
    Parses an NTU RGB+D .skeleton file.
    Returns a numpy array of shape (num_frames, max_bodies, num_joints, 3).
    The 3 coordinates are (x, y, z) in camera space.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    if not lines:
        return np.zeros((0, max_bodies, num_joints, 3))
        
    num_frames = int(lines[0].strip())
    
    # Initialize empty array for this sequence
    data = np.zeros((num_frames, max_bodies, num_joints, 3))
    
    current_line = 1
    for f_idx in range(num_frames):
        if current_line >= len(lines):
            break
            
        num_bodies = int(lines[current_line].strip())
        current_line += 1
        
        for b_idx in range(num_bodies):
            if current_line >= len(lines):
                break
                
            # Body info line (body ID, tracking states, etc.)
            body_info = lines[current_line].strip().split()
            current_line += 1
            
            # Joint count line
            joints_count = int(lines[current_line].strip())
            current_line += 1
            
            for j_idx in range(joints_count):
                if current_line >= len(lines):
                    break
                    
                joint_info = lines[current_line].strip().split()
                # x, y, z are the first 3 floats
                if b_idx < max_bodies and j_idx < num_joints:
                    data[f_idx, b_idx, j_idx, 0] = float(joint_info[0])
                    data[f_idx, b_idx, j_idx, 1] = float(joint_info[1])
                    data[f_idx, b_idx, j_idx, 2] = float(joint_info[2])
                    
                current_line += 1
                
    return data

def center_skeleton(data):
    """
    Centers the skeleton to the origin by subtracting the root joint (usually joint 0 or 1)
    for the primary body in the first frame.
    Data shape: (F, M, V, C) -> Frames, Bodies, Vertices (Joints), Coordinates
    """
    # Assuming joint 1 (Spine base) is the root in NTU (0-indexed)
    root_joint_idx = 0 
    
    # Find first valid frame for body 0
    # For simplicity, just center across all frames using each frame's root
    # Or center the whole sequence using frame 0's root. Let's do per-frame root centering for translation invariance.
    centered_data = np.zeros_like(data)
    for f in range(data.shape[0]):
        for b in range(data.shape[1]):
            # Check if body is present (not all zeros)
            if np.any(data[f, b]):
                root = data[f, b, root_joint_idx, :]
                centered_data[f, b] = data[f, b] - root
                
    return centered_data
