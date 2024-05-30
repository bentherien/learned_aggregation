from learned_optimization.optimizers import base as opt_base
from learned_optimization.outer_trainers import (
    gradient_learner,
    truncated_pes,
    truncation_schedule,
)
from learned_optimization.tasks import base as tasks_base
from learned_optimization.outer_trainers.lopt_truncated_step import VectorizedLOptTruncatedStep
from learned_optimization.learned_optimizers.adafac_mlp_lopt import AdafacMLPLOpt

from fed_adafac_mlp_lopt import FedAdafacMLPLOpt
from fed_truncated_step import VectorizedFedLOptTruncatedStep
from fed_mlp_lopt import FedMLPLOpt
from tasks import get_task
import jax
import optax
from optimizers import AdamWLinearCosine, AdamW
from mup_adafac_mlp_lopt import MuAdafacMLPLOpt


def _fedlagg_meta_trainer(args):
    lagg_class = (
        FedAdafacMLPLOpt
        if args.optimizer in ["fedlopt-adafac", "fedlagg-adafac"]
        else FedMLPLOpt
    )
    with_all_grads = (
        True
        if args.optimizer in ["fedlagg", "fedlagg-wavg", "fedlagg-adafac"]
        else False
    )
    with_avg = (
        True
        if args.optimizer in ["fedlopt", "fedlopt-adafac", "fedlagg-wavg"]
        else False
    )
    lagg = lagg_class(
        num_grads=args.num_grads,
        hidden_size=args.hidden_size,
        with_all_grads=with_all_grads,
        with_avg=with_avg,
    )

    if args.schedule != {}:
        print("Using learning rate scheduler")
        if args.schedule.get("use_adamw", False):
            del args.schedule["use_adamw"]
            meta_opt = AdamW(**args.schedule)
        else:
            meta_opt = AdamWLinearCosine(**args.schedule)
    else:
        meta_opt = opt_base.Adam(args.learning_rate)

    def grad_est_fn(task_family):
        trunc_sched = truncation_schedule.LogUniformLengthSchedule(
            min_length=args.truncation_schedule_min_length, 
            max_length=args.num_inner_steps
        )
        truncated_step = VectorizedFedLOptTruncatedStep(
            task_family=task_family,
            learned_opt=lagg,
            trunc_sched=trunc_sched,
            num_tasks=args.num_tasks,
            meta_loss_split=args.meta_loss_split,
            random_initial_iteration_offset=50,#args.num_inner_steps,
            outer_data_split="train",
            meta_loss_with_aux_key=None,
            local_learning_rate=args.local_learning_rate,
            task_name=task_family.datasets.extra_info['name'],
            num_local_steps=args.num_local_steps,
            keep_batch_in_gpu_memory=args.keep_batch_in_gpu_memory,
        )

        return truncated_pes.TruncatedPES(
            # num_devices=2,
            truncated_step=truncated_step, 
            trunc_length=50,
            std=0.01,
            steps_per_jit=args.steps_per_jit,
            stack_antithetic_samples= False, #default
            sign_delta_loss_scalar= None, #default
        )

    tasks = get_task(args)

    if type(tasks) is list:
        gradient_estimators = [
            grad_est_fn(tasks_base.single_task_to_family(task)) for task in tasks
        ]
    else:
        task_family = tasks_base.single_task_to_family(tasks)
        gradient_estimators = [
            grad_est_fn(task_family),
        ]

    meta_trainer = gradient_learner.SingleMachineGradientLearner(
        meta_init=lagg, 
        gradient_estimators=gradient_estimators, 
        theta_opt=meta_opt, 
        device=jax.local_devices(0)[0]
    )

    return meta_trainer, meta_opt



def _default_meta_trainer(args):
    if 'mup' in args.optimizer:

        lopt = MuAdafacMLPLOpt(exp_mult=0.001,
                            step_mult=args.adafac_step_mult,
                            hidden_size=args.hidden_size,
                            hidden_layers=2,
                            initial_momentum_decays=(0.9, 0.99, 0.999),
                            initial_rms_decays=(0.999,),
                            initial_adafactor_decays=(0.9, 0.99, 0.999),
                            concat_weights=True,
                            make_separate_weights=False,
                            split_weights=False,
                            clip_grad=args.lo_clip_grad,)
                            # mup_lrs=args.runtime_mup_lrs)

    else:
        
        lopt = AdafacMLPLOpt(exp_mult=0.001,
                            step_mult=0.001,
                            hidden_size=args.hidden_size,
                            hidden_layers=2,
                            initial_momentum_decays=(0.9, 0.99, 0.999),
                            initial_rms_decays=(0.999,),
                            initial_adafactor_decays=(0.9, 0.99, 0.999),
                            concat_weights=True,
                            make_separate_weights=False,
                            split_weights=False,
                            clip_grad=args.lo_clip_grad,)

    if args.schedule != {}:
        print("Using learning rate scheduler")
        if args.schedule.get("use_adamw", False):
            del args.schedule["use_adamw"]
            meta_opt = AdamW(**args.schedule)
        else:
            meta_opt = AdamWLinearCosine(**args.schedule)
    else:
        meta_opt = opt_base.Adam(args.learning_rate)

    def grad_est_fn(task_family):
        trunc_sched = truncation_schedule.LogUniformLengthSchedule(
            min_length=args.truncation_schedule_min_length, 
            max_length=args.num_inner_steps
        )
        truncated_step = VectorizedLOptTruncatedStep(
            task_family=task_family,
            learned_opt=lopt,
            trunc_sched=trunc_sched,
            num_tasks=args.num_tasks,
            meta_loss_split=args.meta_loss_split,
            random_initial_iteration_offset=50,#args.num_inner_steps,
            outer_data_split="train",
            meta_loss_with_aux_key=None,
            task_name=task_family.datasets.extra_info['name'],
        )

        return truncated_pes.TruncatedPES(
            # num_devices=2,
            truncated_step=truncated_step, 
            trunc_length=50,
            std=0.01,
            steps_per_jit=args.steps_per_jit,
            stack_antithetic_samples= False, #default
            sign_delta_loss_scalar= None, #default
        )

    tasks = get_task(args)

    if type(tasks) is list:
        gradient_estimators = [
            grad_est_fn(tasks_base.single_task_to_family(task)) for task in tasks
        ]
    else:
        task_family = tasks_base.single_task_to_family(tasks)
        gradient_estimators = [
            grad_est_fn(task_family),
        ]

    meta_trainer = gradient_learner.SingleMachineGradientLearner(
        meta_init=lopt, 
        gradient_estimators=gradient_estimators, 
        theta_opt=meta_opt, 
        device=jax.local_devices(0)[0]
    )

    return meta_trainer, meta_opt


def get_meta_trainer(args):
    meta_trainers = {
        "fedlopt": _fedlagg_meta_trainer,
        "fedlopt-adafac": _fedlagg_meta_trainer,
        "fedlagg": _fedlagg_meta_trainer,
        "fedlagg-wavg": _fedlagg_meta_trainer,
        "fedlagg-adafac": _fedlagg_meta_trainer,
        'small_fc_mlp': _default_meta_trainer,
        'mup_small_fc_mlp': _default_meta_trainer,
    }

    return meta_trainers[args.optimizer](args)  # TODO Find better way to do this
