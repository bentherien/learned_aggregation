import os.path as osp
from tqdm import tqdm
import numpy as np
import jax
import wandb
from learned_optimization import checkpoints

from meta_trainers import get_meta_trainer
import globals


from helpers import get_resume_ckpt, save_checkpoint, set_non_hashable_args

def meta_train(args):
    args = set_non_hashable_args(args)
    meta_trainer, meta_opt = get_meta_trainer(args)

    key = jax.random.PRNGKey(0)
    key, key1 = jax.random.split(key)
    outer_trainer_state = meta_trainer.init(key1)

    globals.needs_state = args.needs_state
    globals.num_grads = args.num_grads
    globals.num_local_steps = args.num_local_steps
    globals.local_batch_size = args.local_batch_size
    globals.use_pmap = args.use_pmap
    globals.num_devices = args.num_devices

    if args.use_pmap:
        assert args.num_grads % args.num_devices == 0, "The number of devices for parallelism should be a divisor of the number of clients (gradients)"

    run = None
    if args.from_checkpoint:
        dirname = osp.join("checkpoints", args.meta_train_name)
        ckpt = open(osp.join(dirname, "latest"), "r").readline().strip()
        outer_trainer_state = checkpoints.load_state(
            osp.join(dirname, "{}.ckpt".format(ckpt)), outer_trainer_state
        )
        run = wandb.init(
            project=args.train_project,
            group=args.meta_train_name,
            config=vars(args),
        )
    elif args.auto_resume:
        ckpt = get_resume_ckpt("checkpoints", args.meta_train_name)
        if ckpt is not None:
            outer_trainer_state = checkpoints.load_state(
                "{}.ckpt".format(ckpt), outer_trainer_state
            )
            run = wandb.init(
                project=args.train_project,
                group=args.meta_train_name,
                config=vars(args),
                resume='must',
                id=ckpt.split('/')[1][:8]
            )
    
    if run == None:
        run = wandb.init(
            project=args.train_project,
            group=args.meta_train_name,
            config=vars(args),
        )

    # import pdb
    # pdb.set_trace()
    # outer_trainer_state.gradient_estimator_states[0].pos_state.inner_opt_state.state

    iteration = int(
        outer_trainer_state.gradient_learner_state.theta_opt_state.iteration
    )
    pbar = tqdm(
        range(iteration, args.num_outer_steps),
        initial=iteration,
        total=args.num_outer_steps,
        ascii=True,
        desc="Outer Loop",
    )
    logging_task_name = args.task[0] if len(args.task) == 1 else "multi-task-with_" + args.task[0]
    for i in pbar:
        key, key1 = jax.random.split(key)
        outer_trainer_state, meta_loss, metrics = meta_trainer.update(
            outer_trainer_state, key1, with_metrics=True
        )

        pbar.set_postfix({
            "Data time":round(np.mean(meta_trainer.gradient_estimators[0].truncated_step.timings[-50 // args.steps_per_jit:]),7),
            "meta loss":round(float(meta_loss),2)
        })

        more_to_log = {
                "iteration": i,
                logging_task_name + " meta loss": meta_loss,
                "learning rate": meta_opt.__dict__.get(
                    "schedule_", lambda x: args.learning_rate
                )(
                    outer_trainer_state.gradient_learner_state.theta_opt_state.iteration
                    - 1
                ),
            }
        
        metrics.update(more_to_log)
        run.log(
            metrics
        )

        if (i + 1) % args.save_iter == 0:  # Checkpoint every 1000th iteration
            save_checkpoint(
                prefix=run.id, i=i, args=args, outer_trainer_state=outer_trainer_state
            )

    savepath = save_checkpoint(
        prefix=run.id, i=i, args=args, outer_trainer_state=outer_trainer_state
    )

    wandb.save(savepath)

    run.finish()
