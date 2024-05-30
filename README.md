# Welcome to $\mu$ Learned Optimization!

# Installation

Run the following code:
```
python -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install nvidia-pyindex
python -m pip install -r requirements.txt
```

# Quickstart - Our quickstart tutorial will be coming soon!

# Meta-training Quickstart
```
CUDA_VISIBLE_DEVICES=0 python src/main.py \
--config config/meta_train/schedules/mxlr=3e-3_mnlr=1e-3_it=5000_clip.py \
--num_tasks 8 \
--local_batch_size 4096 \
--train_project mup-meta-training \
--optimizer mup_small_fc_mlp \
--needs_state \
--steps_per_jit 2 \
--name_suffix _s-mumlp_it=5000_mxlr=3e-3_stepm=01_tasks=8 \
--task mumlp-w128-d3_imagenet-32x32x3 \
--prefetch_batches 20 \
--adafac_step_mult 0.01

CUDA_VISIBLE_DEVICES=0 python src/main.py \
--config config/meta_train/schedules/mxlr=3e-3_mnlr=1e-3_it=5000_clip.py \
--num_tasks 8 \
--local_batch_size 4096 \
--train_project mup-meta-training \
--optimizer mup_small_fc_mlp \
--needs_state \
--steps_per_jit 2 \
--name_suffix _m-mumlp_it=5000_mxlr=3e-3_stepm=01_tasks=8 \
--prefetch_batches 20 \
--adafac_step_mult 0.01 \
--task mumlp-w1024-d3_imagenet-32x32x3,mumlp-w512-d3_imagenet-32x32x3,mumlp-w128-d3_imagenet-32x32x3 \
--auto_resume
```




# Testing Quickstart


## Test VeLO
```
CUDA_VISIBLE_DEVICES=0 python src/main.py \
--config config/meta_test/image-mlp-fmst_fedlagg-adafac.py \
--name_suffix _m_mup_final \
--local_batch_size 128 \
--test_project mup-meta-testing \
--task mutransformer-w2048-d3_lm1b-s64-v32k \
--optimizer mup_small_fc_mlp \
--wandb_checkpoint_id eb-lab/mup-meta-training/woz3g9l0 \
--num_runs 5 \
--num_inner_steps 5000 \
--needs_state \
--adafac_step_mult 0.01 \
--gradient_accumulation_steps 1 \
--test_interval 100
```

## Test MuLO
```
CUDA_VISIBLE_DEVICES=0 python src/main.py \
--config config/meta_test/image-mlp-fmst_fedlagg-adafac.py \
--name_suffix _m_mup_final \
--local_batch_size 128 \
--test_project mup-meta-testing \
--task mutransformer-w2048-d3_lm1b-s64-v32k \
--optimizer mup_small_fc_mlp \
--wandb_checkpoint_id <WANDB PATH TO CHECKPOINT> \
--num_runs 5 \
--num_inner_steps 5000 \
--needs_state \
--adafac_step_mult 0.01 \
--gradient_accumulation_steps 1 \
--test_interval 100 \
--use_bf16
```

## Sweep MuAdam Leerning Rates
```
CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --config config/sweeps/sweep_muadam.py \
    --name_suffix _muadam_sweep \
    --local_batch_size 4096 \
    --test_project mup-meta-testing \
    --task "mumlp-w1024-d3_imagenet-32x32x3" \
    --optimizer muadam \
    --num_runs 1 \
    --learning_rate 3e-4 \
    --num_inner_steps 1000 \
    --gradient_accumulation_steps 1 \
    --needs_state \
    --mup_input_mult 1 \
    --mup_output_mult 1 \
    --mup_hidden_lr_mult 1 \
    --test_interval 50 \
```

```

# Config file structure

Using MMengine's config file parser, we can write config files directly in Python and use an inheritance config structure to avoid redundant configurations. This can be achieved by specifying config files to inherit from using the 
```_base_=['my_config.py']``` 
special variable at the top of config files. More information is available at [mmengine config docs](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html).

In learned_aggragation, configuration files are logically separated into different directories based on the task to be executed: ```config/meta_test```,```config/meta_train```, and ```config/sweeps```. 

# Setting up a sweep
To sweep over the hyperparameters of a model during meta-testing, one can simply specify a sweep configuration using the ```sweep_config``` variable.


# Checkpointing during meta training
The ```checkpoints_to_keep``` and ```save_iter``` config variables control the number of checkpoints that should be kept and the checkpointing multiple, respectively. Default values of ```checkpoints_to_keep=10``` and ```save_iter=1000``` ensure that at most 10 previous checkpoints will be kept and that a checkpoint will be saved every 1000 iterations.

# Loading from a checkpoint during meta training
When a checkpoint is logged, it is saved under ```checkpoints/<meta-train-dir>``` where ```<meta-train-dir>``` is the dynamically assigned meta-train-name. Whenever a new checkpoint is logged, a file called ```latest``` is updated with the name of the most recent checkpoint. When resuming from a checkpoint the user simply has to set the ```--from_checkpoint``` flag and meta training will automatically resume to the checkpoint specified in the ```latest``` file.
