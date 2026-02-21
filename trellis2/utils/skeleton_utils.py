"""
SMPL 22-joint skeleton definition and mapping utilities.

Provides canonical joint names, parent-child connectivity, and helpers
for building joint index mappings between different skeleton conventions
(e.g. UniRig output -> SMPL for motion retargeting in Phase 2).
"""

SMPL_22_JOINTS = [
    "pelvis",          # 0
    "left_hip",        # 1
    "right_hip",       # 2
    "spine1",          # 3
    "left_knee",       # 4
    "right_knee",      # 5
    "spine2",          # 6
    "left_ankle",      # 7
    "right_ankle",     # 8
    "spine3",          # 9
    "left_foot",       # 10
    "right_foot",      # 11
    "neck",            # 12
    "left_collar",     # 13
    "right_collar",    # 14
    "head",            # 15
    "left_shoulder",   # 16
    "right_shoulder",  # 17
    "left_elbow",      # 18
    "right_elbow",     # 19
    "left_wrist",      # 20
    "right_wrist",     # 21
]

# Parent index for each joint (-1 = root)
SMPL_22_PARENTS = [
    -1,  # pelvis
     0,  # left_hip -> pelvis
     0,  # right_hip -> pelvis
     0,  # spine1 -> pelvis
     1,  # left_knee -> left_hip
     2,  # right_knee -> right_hip
     3,  # spine2 -> spine1
     4,  # left_ankle -> left_knee
     5,  # right_ankle -> right_knee
     6,  # spine3 -> spine2
     7,  # left_foot -> left_ankle
     8,  # right_foot -> right_ankle
     9,  # neck -> spine3
     9,  # left_collar -> spine3
     9,  # right_collar -> spine3
    12,  # head -> neck
    13,  # left_shoulder -> left_collar
    14,  # right_shoulder -> right_collar
    16,  # left_elbow -> left_shoulder
    17,  # right_elbow -> right_shoulder
    18,  # left_wrist -> left_elbow
    19,  # right_wrist -> right_elbow
]

# (parent_idx, child_idx) pairs for all bones
SMPL_22_CONNECTIONS = [
    (parent, child)
    for child, parent in enumerate(SMPL_22_PARENTS)
    if parent >= 0
]


def build_joint_mapping(source_joints, target_joints, custom_map=None):
    """Build a source -> target joint index mapping.

    Args:
        source_joints: List of joint name strings from the source skeleton.
        target_joints: List of joint name strings from the target skeleton.
        custom_map: Optional dict of {source_name: target_name} overrides
                    for joints whose names don't match exactly.

    Returns:
        Dict mapping source joint index -> target joint index.
        Only includes joints that could be matched.
    """
    custom_map = custom_map or {}

    target_lookup = {name.lower(): idx for idx, name in enumerate(target_joints)}
    mapping = {}

    for src_idx, src_name in enumerate(source_joints):
        # Check custom override first
        if src_name in custom_map:
            tgt_name = custom_map[src_name].lower()
        else:
            tgt_name = src_name.lower()

        if tgt_name in target_lookup:
            mapping[src_idx] = target_lookup[tgt_name]

    return mapping
