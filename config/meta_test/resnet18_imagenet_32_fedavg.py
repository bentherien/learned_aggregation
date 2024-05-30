_base_ = ["./meta_test_base.py"]

optimizer = "fedavg"
task = "resnet18_imagenet_32"
needs_state = True

# values determined by sweep
local_learning_rate = 1
