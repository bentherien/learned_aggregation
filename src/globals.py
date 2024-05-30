# Bad practice but the learned_optimization code is so nested that this is probably the easiest way to implement changes

needs_state = False
num_grads = 2
num_local_steps = 4
local_batch_size = 128
use_pmap = True
num_devices = 2