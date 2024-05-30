import argparse

import jax
import jax.numpy as jnp
import wandb
from tqdm import tqdm

import pprint
from optimizers import get_optimizer
from tasks import get_task
import globals
import time
from functools import reduce, partial
import numpy as np

from helpers import set_non_hashable_args

def cast_to_bf16(pytree):
    """
    Recursively cast all JAX arrays within a PyTree to BF16.
    """
    return jax.tree_map(lambda x: x.astype(jnp.bfloat16) if isinstance(x, jnp.ndarray) else x, pytree)


is_leaf = lambda x : reduce(np.logical_and, [type(x1) != dict for x1 in x.values()])



def add_prefix(prefix,s):
    if prefix != '':
        prefix = prefix + '/'
    return prefix + s

def get_mup_lrs(state,prefix):
    d = {}
    for k,v in state.items():
        if is_leaf(v):
            d[add_prefix(prefix,k)] = v
        else:
            for kk,vv in get_mup_lrs(v,k).items():
                d[add_prefix(prefix,kk)] = vv
    
    d = {k.replace('/mup_lrs',''):v for k,v in d.items()}
    return d
# lrs = get_mup_lrs({k:{'mup_lrs':v['mup_lrs']} for k,v in state.items() if 'mup_lrs'in v.keys()}, 
#                         prefix='')

def rename_batch(batch):
    label_map = {'obs':'image',
                    'target':'label',
                    'image':'image',
                    'label':'label'}
    
    return {label_map[k]:v for k,v in batch.items()}

def count_parameters(params):
    return sum(jnp.size(param) for param in jax.tree_util.tree_leaves(params))

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key+'_mean', jnp.mean(v).item()))
            items.append((new_key+'_std', jnp.std(v).item()))
            items.append((new_key+'_max', jnp.max(v).item()))
            items.append((new_key+'_min', jnp.min(v).item()))
            items.append((new_key+'_2norm', jnp.linalg.norm(v,ord=2).item()))

    return dict(items)

def get_params_and_state(needs_state, task, key):
    if needs_state:
        return task.init_with_state(key)
    else:
        return task.init(key), None

class Timing:
    def __init__(self,name,list):
        self.name = name
        self.list = list

    def __enter__(self):
        self.start = time.time()
        return self  # This allows us to use "as x" in the with statement

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()
        self.duration = self.end - self.start
        self.list.append(self.duration)
        # print(f"[{self.name}] Block took {self.duration:.6f} seconds")

def benchmark(args, sweep=False):
    if sweep:
        run = wandb.init(project=args.test_project, group=args.name, config=vars(args))   
        args = argparse.Namespace(**run.config)
    
    args = set_non_hashable_args(args)
    # Set up globals used in truncated step for benchmarking
    globals.needs_state = args.needs_state
    globals.num_grads = args.num_grads
    globals.num_local_steps = args.num_local_steps
    globals.local_batch_size = args.local_batch_size
    globals.use_pmap = args.use_pmap
    globals.num_devices = args.num_devices

    key = jax.random.PRNGKey(args.seed)
    task = get_task(args)
    # test_task = get_task(args, is_test=True)

    key, key1 = jax.random.split(key)
    params, state = get_params_and_state(args.needs_state, task, key1)

    if args.use_bf16:
        params = cast_to_bf16(params)

    print("Model parameters (M): ", count_parameters(params)/1e6)
    
    if state is not None:
        try:
            lrs = state['mup_lrs_to_use']
            assert len(lrs) == len(params), "Number of learning rates should be equal to number of parameters"
            assert set(lrs.keys()) == set(params.keys()), "Learning rates should have the same keys as parameters"
            args.runtime_mup_lrs = lrs
        except KeyError as e:
            print("No mup_lrs_to_use in state, for task "+args.task[0])


    opt, update = get_optimizer(args)

    if args.use_pmap:
        assert args.num_grads % args.num_devices == 0, "The number of devices for pmap should be a multiple of the number of clients (gradients)"


    
    
    print('\nstarting loop')
    for _ in tqdm(range(args.num_runs), ascii=True, desc="Outer Loop"):
        if not sweep:
            run = wandb.init(project=args.test_project, group=args.name, config=vars(args))
        
        if _ > 0:
            params, state = get_params_and_state(args.needs_state, task, key1)
        
        opt_state = opt.init(params, model_state=state, num_steps=args.num_inner_steps)
        prev_params = params

        pbar = tqdm(
            range(args.num_inner_steps),
            initial=0,
            total=args.num_inner_steps,
            ascii=True,
            desc="Inner Loop",
        )
        train_loadl, gradl, stepl, testl = [],[],[],[]
        for iteration in pbar:

            # update
            with Timing('get traing batch',train_loadl):
                batch = rename_batch(next(task.datasets.train))
                if args.use_bf16:
                    batch = cast_to_bf16(batch)

            key, key1 = jax.random.split(key)
            # print('in benchmark',jax.tree_map(lambda x: x.shape, batch))

            with Timing('fw bw',gradl):
                opt_state, loss, grad = update(opt_state, key1, batch)
                to_log = {
                        "train loss": loss,
                    }

            with Timing('opt',stepl):
                params = opt.get_params(opt_state)
                if args.use_bf16:
                    params = cast_to_bf16(params)


            with Timing('test',testl):
                #test loss and accuracy if implemented
                if not args.skip_test and iteration % args.test_interval == 0 or iteration == 8:
                    try:
                        test_batch = rename_batch(next(task.datasets.test))
                        if args.use_bf16:
                            test_batch = cast_to_bf16(test_batch)
                        key, key1 = jax.random.split(key)

                        if args.needs_state:
                            state = opt.get_state(opt_state)
                            test_loss, test_acc = task.loss_and_accuracy_with_state(params, state, key1, test_batch)
                        else:
                            test_loss, test_acc = task.loss_and_accuracy(params, key1, test_batch)
                        test_log = {
                            "test loss": test_loss,
                            "test accuracy": test_acc,
                        }
                    except AttributeError as e:
                        Warning("test_task does not have loss_and_accuracy method, defaulting to loss")
                        key, key1 = jax.random.split(key)
                        if args.needs_state:
                            state = opt.get_state(opt_state)
                            test_loss, state = task.loss_with_state(params, state, key1, test_batch)
                        else:
                            test_loss = task.loss(params, key1, test_batch)

                        test_log = {"test loss": test_loss}
                    
                    to_log.update(test_log)
                else:
                    test_loss = 0

            # valid loss
            # outer_valid_batch = rename_batch(next(test_task.datasets.outer_valid))
            # key, key1 = jax.random.split(key)
            # if args.needs_state:
            #     state = opt.get_state(opt_state)
            #     outer_valid_loss = test_task.loss(params, state, key1, outer_valid_batch)
            # else:
            #     outer_valid_loss = test_task.loss(params, key1, outer_valid_batch)

            
            pbar.set_postfix({
                    "Data time train":round(train_loadl[-1],4),
                    "fwbw time":round(gradl[-1],4),
                    "opt time":round(stepl[-1],4),
                    "test time":round(testl[-1],4),
                    "train loss":round(float(loss),2),
                    "test loss":round(float(test_loss),2) if not args.skip_test else 0
                })

            # log

            # to_log.update(flatten_dict(grad, parent_key='', sep='_'))
            # to_log.update(flatten_dict(jax.tree_map(lambda x,y:x-y,prev_params,params), parent_key='delta', sep='_'))
            run.log(to_log)

            prev_params = params


        run.finish()


def sweep(args):
    def sweep_fn(args=args):
        run = wandb.init(
            project="learned_aggregation_meta_test", group=args.name, config=vars(args)
        )
        args = argparse.Namespace(**run.config)

        key = jax.random.PRNGKey(0)
        task = get_task(args)
        test_task = get_task(args, is_test=True)
        opt, update = get_optimizer(args)

        key, key1 = jax.random.split(key)
        if args.needs_state:
            params, state = task.init_with_state(key1)
        else:
            params, state = task.init(key1), None
        
        opt_state = opt.init(params, model_state=state, num_steps=args.num_inner_steps)

        for _ in tqdm(range(args.num_inner_steps), ascii=True, desc="Inner Loop"):
            # update
            batch = rename_batch(next(task.datasets.train))
            key, key1 = jax.random.split(key)
            opt_state, loss = update(opt_state, key1, batch)
            params = opt.get_params(opt_state)

            #test loss and accuracy if implemented
            try:
                test_batch = rename_batch(next(test_task.datasets.test))
                key, key1 = jax.random.split(key)
                test_loss, test_acc = test_task.loss_and_accuracy(params, key1, test_batch)
                test_log = {
                    "test loss": test_loss,
                    "test accuracy": test_acc,
                }
            except AttributeError as e:
                Warning("test_task does not have loss_and_accuracy method, defaulting to loss")
                key, key1 = jax.random.split(key)
                if args.needs_state:
                    state = opt.get_state(opt_state)
                    test_loss = test_task.loss(params, state, key1, test_batch)
                else:
                    test_loss = test_task.loss(params, key1, test_batch)

                test_log = {"test loss": test_loss}

            # valid loss
            outer_valid_batch = rename_batch(next(test_task.datasets.outer_valid))
            key, key1 = jax.random.split(key)
            if args.needs_state:
                state = opt.get_state(opt_state)
                outer_valid_loss = test_task.loss(params, state, key1, outer_valid_batch)
            else:
                outer_valid_loss = test_task.loss(params, key1, outer_valid_batch)
            
            # log
            to_log = {
                    "train loss": loss,
                    "outer valid loss": outer_valid_loss
                }
            to_log.update(test_log)
            run.log(to_log)

        run.finish()

    # if args.sweep_id is None:
    #     args.sweep_id = wandb.sweep(
    #         sweep=args.sweep_config, project="learned_aggregation_meta_test"
    #     )
    import os
    os.environ['WANDB_LOG_LEVEL'] = 'debug'
    # wandb.agent(args.sweep_id, sweep_fn, project="learned_aggregation_meta_test")
    for k,v in args.__dict__.items():
        if type(v) == list:
            print(k,type(v))

    print(args.sweep_config)
    if args.sweep_id is None:
        args.sweep_id = wandb.sweep(
            sweep=args.sweep_config, project="mup-meta-testing"
        )

    print('\n[info] in sweep before creating agent')
    wandb.agent(args.sweep_id, partial(benchmark, args, True), project="mup-meta-testing")