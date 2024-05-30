import subprocess
import os
import time

def run_commands_on_gpus(commands, GPUS):

    num_gpus = len(GPUS)

    # Current GPU index
    gpu_index = 0

    # Process list to keep track of running processes
    processes = []

    # Assign each command to a GPU
    for command in commands:
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(GPUS[gpu_index])

        # Print which GPU is being used (for debugging purposes)
        print(f"Running command on GPU {gpu_index}: {command}")

        # Start the command
        process = subprocess.Popen(command, shell=True, env=env)
        processes.append(process)

        # Move to the next GPU
        gpu_index = (gpu_index + 1) % num_gpus

        # Wait for a GPU to free up if all are used
        if len(processes) >= num_gpus:
            # Wait for the first process to complete
            processes[0].wait()
            # Remove the completed process
            processes.pop(0)

    # Wait for all remaining processes to complete
    for process in processes:
        process.wait()

# Example usage:
commands =['python src/main.py         --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py         --name_suffix _s_mup_final         --local_batch_size 128         --test_project mup-meta-testing         --task mutransformer-w1024-d3_lm1b-s64-v32k         --optimizer mup_small_fc_mlp         --wandb_checkpoint_id eb-lab/mup-meta-training/2tfh8kuw         --num_runs 5         --num_inner_steps 5000         --needs_state         --adafac_step_mult 0.01         --gradient_accumulation_steps 1         --test_interval 100',
 'python src/main.py         --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py         --name_suffix _m_mup_final         --local_batch_size 128         --test_project mup-meta-testing         --task mutransformer-w1024-d3_lm1b-s64-v32k         --optimizer mup_small_fc_mlp         --wandb_checkpoint_id eb-lab/mup-meta-training/woz3g9l0         --num_runs 5         --num_inner_steps 5000         --needs_state         --adafac_step_mult 0.01         --gradient_accumulation_steps 1         --test_interval 100',
 'python src/main.py         --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py         --name_suffix _velo_final         --local_batch_size 128         --test_project mup-meta-testing         --task transformer-w1024-d3_lm1b-s64-v32k         --optimizer velo         --wandb_checkpoint_id eb-lab/mup-meta-training/2tfh8kuw         --num_runs 5         --num_inner_steps 5000         --needs_state         --adafac_step_mult 0.001         --gradient_accumulation_steps 1         --test_interval 100',
 'python src/main.py         --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py         --name_suffix _s_sp_final         --local_batch_size 128         --test_project mup-meta-testing         --task transformer-w1024-d3_lm1b-s64-v32k         --optimizer small_fc_mlp         --wandb_checkpoint_id eb-lab/mup-meta-training/dcjjqqyt         --num_runs 5         --num_inner_steps 5000         --needs_state         --adafac_step_mult 0.001         --gradient_accumulation_steps 1         --test_interval 100',
 'python src/main.py         --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py         --name_suffix _m_sp_final         --local_batch_size 128         --test_project mup-meta-testing         --task transformer-w1024-d3_lm1b-s64-v32k         --optimizer small_fc_mlp         --wandb_checkpoint_id eb-lab/mup-meta-training/byuo0ixg         --num_runs 5         --num_inner_steps 5000         --needs_state         --adafac_step_mult 0.001         --gradient_accumulation_steps 1         --test_interval 100',
 'python src/main.py         --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py         --name_suffix _s_mup_final         --local_batch_size 128         --test_project mup-meta-testing         --task mutransformer-w192-d3_lm1b-s64-v32k         --optimizer mup_small_fc_mlp         --wandb_checkpoint_id eb-lab/mup-meta-training/2tfh8kuw         --num_runs 5         --num_inner_steps 5000         --needs_state         --adafac_step_mult 0.01         --gradient_accumulation_steps 1         --test_interval 100',
 'python src/main.py         --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py         --name_suffix _m_mup_final         --local_batch_size 128         --test_project mup-meta-testing         --task mutransformer-w192-d3_lm1b-s64-v32k         --optimizer mup_small_fc_mlp         --wandb_checkpoint_id eb-lab/mup-meta-training/woz3g9l0         --num_runs 5         --num_inner_steps 5000         --needs_state         --adafac_step_mult 0.01         --gradient_accumulation_steps 1         --test_interval 100',
 'python src/main.py         --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py         --name_suffix _velo_final         --local_batch_size 128         --test_project mup-meta-testing         --task transformer-w192-d3_lm1b-s64-v32k         --optimizer velo         --wandb_checkpoint_id eb-lab/mup-meta-training/2tfh8kuw         --num_runs 5         --num_inner_steps 5000         --needs_state         --adafac_step_mult 0.001         --gradient_accumulation_steps 1         --test_interval 100',
 'python src/main.py         --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py         --name_suffix _s_sp_final         --local_batch_size 128         --test_project mup-meta-testing         --task transformer-w192-d3_lm1b-s64-v32k         --optimizer small_fc_mlp         --wandb_checkpoint_id eb-lab/mup-meta-training/dcjjqqyt         --num_runs 5         --num_inner_steps 5000         --needs_state         --adafac_step_mult 0.001         --gradient_accumulation_steps 1         --test_interval 100',
 'python src/main.py         --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py         --name_suffix _m_sp_final         --local_batch_size 128         --test_project mup-meta-testing         --task transformer-w192-d3_lm1b-s64-v32k         --optimizer small_fc_mlp         --wandb_checkpoint_id eb-lab/mup-meta-training/byuo0ixg         --num_runs 5         --num_inner_steps 5000         --needs_state         --adafac_step_mult 0.001         --gradient_accumulation_steps 1         --test_interval 100'
]

GPUS = [0]
run_commands_on_gpus(commands, GPUS=GPUS)