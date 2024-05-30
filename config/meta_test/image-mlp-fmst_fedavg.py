_base_ = ["./meta_test_base.py"]

optimizer = "fedavg"
task = "image-mlp-fmst"

# value determined by sweep
local_learning_rate = 0.3
