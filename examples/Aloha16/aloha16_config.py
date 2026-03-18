from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

papras_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["base_rgb", "left_wrist_rgb", "right_wrist_rgb"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm_joints",
            "left_gripper",
            "right_arm_joints",
            "right_gripper",
        ],
        sin_cos_embedding_keys=["left_arm_joints", "right_arm_joints"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=[
            "left_arm_joints",
            "left_gripper",
            "right_arm_joints",
            "right_gripper",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="left_arm_joints",
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="left_gripper",
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="right_arm_joints",
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="right_gripper",
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=[f'task'],
    ),
}

register_modality_config(papras_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
