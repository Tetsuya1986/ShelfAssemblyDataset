import numpy as np

# Define a kinematic tree for the skeletal struture
kit_kinematic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]]

kit_raw_offsets = np.array(
    [
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [-1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, 1]
    ]
)

t2m_raw_offsets = np.array([[0,0,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,0,1],
                           [0,0,1],
                           [0,1,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,0,1],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0]])

# t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
# t2m_left_hand_chain = [[20, 22, 23, 24], [20, 34, 35, 36], [20, 25, 26, 27], [20, 31, 32, 33], [20, 28, 29, 30]]
# t2m_right_hand_chain = [[21, 43, 44, 45], [21, 46, 47, 48], [21, 40, 41, 42], [21, 37, 38, 39], [21, 49, 50, 51]]

# for shelf assembly dataset (SMPL-X 53/127 joints)
# Body (0-21)
# 0: Pelvis, 1: L_Hip, 2: R_Hip, 3: Spine1, 4: L_Knee, 5: R_Knee, 6: Spine2, 7: L_Ankle, 8: R_Ankle, 9: Spine3
# 10: L_Foot, 11: R_Foot, 12: Neck, 13: L_Collar, 14: R_Collar, 15: Head, 16: L_Shoulder, 17: R_Shoulder
# 18: L_Elbow, 19: R_Elbow, 20: L_Wrist, 21: R_Wrist
t2m_kinematic_chain = [
    [0, 2, 5, 8, 11],       # Right Leg
    [0, 1, 4, 7, 10],       # Left Leg
    [0, 3, 6, 9, 12, 15],   # Spine
    [9, 14, 17, 19, 21],    # Right Arm (Spine3 -> Collar -> Shoulder -> Elbow -> Wrist)
    [9, 13, 16, 18, 20]     # Left Arm (Spine3 -> Collar -> Shoulder -> Elbow -> Wrist)
]

# Left Hand (25-39) connected to Wrist (20)
# Index: 20 -> 25 -> 26 -> 27
# Middle: 20 -> 28 -> 29 -> 30
# Ring: 20 -> 31 -> 32 -> 33
# Pinky: 20 -> 34 -> 35 -> 36
# Thumb: 20 -> 37 -> 38 -> 39
t2m_left_hand_chain = [
    [20, 25, 26, 27], [20, 28, 29, 30], [20, 31, 32, 33], [20, 34, 35, 36], [20, 37, 38, 39]
]

# Right Hand (40-54) connected to Wrist (21)
# Index: 21 -> 40 -> 41 -> 42
# Middle: 21 -> 43 -> 44 -> 45
# Ring: 21 -> 46 -> 47 -> 48
# Pinky: 21 -> 49 -> 50 -> 51
# Thumb: 21 -> 52 -> 53 -> 54
t2m_right_hand_chain = [
    [21, 40, 41, 42], [21, 43, 44, 45], [21, 46, 47, 48], [21, 49, 50, 51], [21, 52, 53, 54]
]

kit_tgt_skel_id = '03950'

t2m_tgt_skel_id = '000021'

