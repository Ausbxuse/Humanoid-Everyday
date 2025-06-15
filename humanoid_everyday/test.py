from humanoid_everyday import Dataloader

# Load your downloaded task's dataset zip file (e.g., the "push_a_button" task)
ds = Dataloader("~/Downloads/flip_a_soft_plate.zip")
print("Episode length of dataset:", len(ds))

# Displaying high dimensional data at first episode, second timestep.
ds.display_image(0, 1)
ds.display_depth_point_cloud(0, 1)
ds.display_lidar_point_cloud(0, 1)
for i, episode in enumerate(ds):
    if i == 1:  # episode 1
        print("RGB image shape:", episode[0]["image"].shape)  # (480, 640, 3)
        print("Depth map shape:", episode[0]["depth"].shape)  # (480, 640)
        print("LiDAR points shape:", episode[0]["lidar"].shape)  # (~6000, 3)

        batch = episode[0:4]  # batch loading episodes
        print(batch[1]["image"].shape)
        print(batch[0]["image"].shape)
