from humanoid_everyday import Dataloader
import numpy as np

# Load your downloaded task's dataset zip file (e.g., the "push_a_button" task)
ds = Dataloader("~/pick and place orange duster.zip")
print("Episode length of dataset:", len(ds))

# Display high dimensional data at first episode, second timestep.
ds.display_image(0, 1)
ds.display_depth_point_cloud(0, 1)
ds.display_lidar_point_cloud(0, 1)
for i, episode in enumerate(ds):
    if i == 1:  # episode 1
        robot_type = "G1" if "robot_type" in episode[0] else "H1"
        print("Robot type:", robot_type)
        states = np.array(episode[0]["states"]["arm_state"] + episode[0]["states"]["hand_state"])
        print("States shape:", states.shape) # 26 for H1 and 28 for G1

        if robot_type == "H1":  # We recorded internal hand representation for H1 hand data, which needs post-processing
            left_qpos = episode[0]["actions"]["left_angles"]
            right_qpos = episode[0]["actions"]["right_angles"]

            right_hand_angles = [1.7 - right_qpos[i] for i in [4, 6, 2, 0]]
            right_hand_angles.append(1.2 - right_qpos[8])
            right_hand_angles.append(0.5 - right_qpos[9])
            left_hand_angles = [1.7 - left_qpos[i] for i in [4, 6, 2, 0]]
            left_hand_angles.append(1.2 - left_qpos[8])
            left_hand_angles.append(0.5 - left_qpos[9])

            hand_actions = left_hand_angles + right_hand_angles # 12-dim H1 hand actions
        else:   # For G1 data
            hand_actions = episode[0]["actions"]["left_angles"] + episode[0]["actions"]["right_angles"] # 14-dim G1 hand actions
        
        arm_actions = episode[0]["actions"]["sol_q"] # 14-dim arm actions
        actions = np.array(arm_actions + hand_actions)
        print("Actions shape:", actions.shape) # 26 for H1 and 28 for G1
        
        print("RGB image shape:", episode[0]["image"].shape)  # (480, 640, 3)
        print("Depth map shape:", episode[0]["depth"].shape)  # (480, 640)
        print("LiDAR points shape:", episode[0]["lidar"].shape)  # (~6000, 3)

        batch = episode[0:4]  # batch loading episodes
        print(batch[1]["image"].shape)
        print(batch[0]["image"].shape)
