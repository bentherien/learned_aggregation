# must be overriden
run_type = None

# common
optimizer = "fedavg"
task = "small-image-mlp-fmst"
hidden_size = 32
local_learning_rate = 0.5
local_batch_size = 128
num_grads = 8
num_local_steps = 4
num_inner_steps = 1000
learning_rate = 0.0001
name_suffix = ""
needs_state = False # For models storing inside state such as resnets
num_devices = 1

# meta training only
num_outer_steps = 5000
from_checkpoint = False
use_pmap = False
auto_resume = False
meta_loss_split = None
steps_per_jit = 10
num_tasks = 8
train_project = "learned_aggregation_meta_train"
prefetch_batches = 20

# meta testing only
num_runs = 10
wandb_checkpoint_id = None
gradient_accumulation_steps = 1
test_project = "learned_aggregation_meta_test"
truncation_schedule_min_length = 100
# for slowmo
beta = 0.99
test_interval = 50
adafac_step_mult=0.001

seed=0


#MuP
mup_input_mult = 1.0
mup_output_mult = 1.0
mup_hidden_lr_mult = 1.0


# sweeps only
sweep_config = dict()
sweep_id = None



keep_batch_in_gpu_memory = False