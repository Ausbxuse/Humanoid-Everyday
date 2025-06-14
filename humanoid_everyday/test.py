from humanoid_everyday import Dataloader

a = Dataloader("~/Downloads/push_a_button.zip").data
print(a[0][0]["image"].shape)
print(a[0][0]["depth"].shape)
