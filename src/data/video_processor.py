import cv2
import numpy as np
import mediapipe as mp
import torch

class MediaPipePoseExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def _map_to_ntu(self, landmarks):
        """
        Map 33 MediaPipe landmarks to 25 NTU RGB+D joints.
        Returns a numpy array of shape (25, 3).
        """
        ntu_joints = np.zeros((25, 3))
        
        # 33 landmarks, each with x, y, z
        lm = np.array([[l.x, l.y, l.z] for l in landmarks.landmark])
        
        # NTU 25 Joints Mapping
        # 0: spine base (pelvis) -> avg(left_hip(23), right_hip(24))
        ntu_joints[0] = (lm[23] + lm[24]) / 2.0
        
        # 20: spine shoulder -> avg(left_shoulder(11), right_shoulder(12))
        ntu_joints[20] = (lm[11] + lm[12]) / 2.0
        
        # 1: spine mid -> avg(spine_base, spine_shoulder)
        ntu_joints[1] = (ntu_joints[0] + ntu_joints[20]) / 2.0
        
        # 2: neck -> avg(spine_shoulder, nose(0))
        ntu_joints[2] = (ntu_joints[20] + lm[0]) / 2.0
        
        # 3: head -> nose(0)
        ntu_joints[3] = lm[0]
        
        # Arms
        ntu_joints[4] = lm[11]  # shoulder left
        ntu_joints[5] = lm[13]  # elbow left
        ntu_joints[6] = lm[15]  # wrist left
        ntu_joints[7] = lm[19]  # hand left (index)
        
        ntu_joints[8] = lm[12]  # shoulder right
        ntu_joints[9] = lm[14]  # elbow right
        ntu_joints[10] = lm[16] # wrist right
        ntu_joints[11] = lm[20] # hand right (index)
        
        # Legs
        ntu_joints[12] = lm[23] # hip left
        ntu_joints[13] = lm[25] # knee left
        ntu_joints[14] = lm[27] # ankle left
        ntu_joints[15] = lm[31] # foot left (foot index)
        
        ntu_joints[16] = lm[24] # hip right
        ntu_joints[17] = lm[26] # knee right
        ntu_joints[18] = lm[28] # ankle right
        ntu_joints[19] = lm[32] # foot right (foot index)
        
        # Hands tips/thumbs
        ntu_joints[21] = lm[17] # hand tip left (pinky)
        ntu_joints[22] = lm[21] # thumb left
        ntu_joints[23] = lm[18] # hand tip right (pinky)
        ntu_joints[24] = lm[22] # thumb right
        
        return ntu_joints

    def extract_from_video(self, video_path):
        """
        Extract skeleton frames from video.
        Returns array of shape (num_frames, 25, 3)
        """
        cap = cv2.VideoCapture(video_path)
        frames_skeleton = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            results = self.pose.process(image)
            
            if results.pose_world_landmarks:
                # Use world landmarks for true 3D in meters (if origin is pelvis roughly)
                joints = self._map_to_ntu(results.pose_world_landmarks)
                frames_skeleton.append(joints)
            else:
                # Fallback to zero if no person detected
                frames_skeleton.append(np.zeros((25, 3)))
                
        cap.release()
        
        if not frames_skeleton:
            return np.zeros((1, 25, 3))
            
        return np.array(frames_skeleton)
