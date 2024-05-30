import functools
import pickle

import gin
import jax
import jax.numpy as jnp
import optax
from learned_optimization.optimizers import OptaxOptimizer
from learned_optimization.optimizers import base as opt_base
from learned_optimization.optimizers import optax_opts

from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map

from fed_adafac_mlp_lopt import FedAdafacMLPLOpt
from fed_mlp_lopt import FedMLPLOpt
from slowmo import SGDSlowMo
from tasks import get_task
import globals
from learned_optimization.learned_optimizers.adafac_mlp_lopt import AdafacMLPLOpt
from learned_optimization.research.general_lopt import prefab
from mup_adafac_mlp_lopt import MuAdafacMLPLOpt
import mmengine

@gin.configurable
class AdamWLinearCosine(OptaxOptimizer):
    """Adam with a piecewise linear learning rate schedule."""

    def __init__(
        self,
        init_value=3e-10,
        peak_value=3e-4,
        warmup_steps=300,
        decay_steps=9700,
        end_value=3e-5,
        exponent=1.0,
        clip=False,
    ):
        self.schedule_ = optax.warmup_cosine_decay_schedule(
            init_value=init_value,
            peak_value=peak_value,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=end_value,
            exponent=exponent,
        )
        if clip:
            opt = optax.chain(
                optax.adamw(self.schedule_),
                optax.clip_by_global_norm(1.0),
            )
        else:
            opt = optax.adamw(self.schedule_)

        super().__init__(opt)

@gin.configurable
class MuAdamWLinearCosine(OptaxOptimizer):
    """Adam with a piecewise linear learning rate schedule."""

    def __init__(
        self,
        init_value=3e-10,
        peak_value=3e-4,
        warmup_steps=300,
        decay_steps=9700,
        end_value=3e-5,
        exponent=1.0,
        clip=1.0,
        mup_lrs=None,
    ):
        assert mup_lrs is not None, "must provide mup_lrs"

        self.schedule_ = optax.warmup_cosine_decay_schedule(
            init_value=init_value,
            peak_value=peak_value,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=end_value,
            exponent=exponent,
        )

        def init_fn(params):
            del params
            return optax.EmptyState()
        
        def update_fn(updates, state, params=None):
            del params
            # print(updates)
            # print(mup_lrs)
            # import pdb; pdb.set_trace()
            updates = jax.tree_map(
                lambda update, scale: update * scale,
                updates,
                mup_lrs
            )
            # jax.tree_map(lambda update, scale: update * scale, updates, dict(mup_lrs))
            return updates, state
        
        opt = optax.chain(
            optax.adamw(self.schedule_),
            optax.GradientTransformation(init_fn, update_fn),
            optax.clip_by_global_norm(clip),
        )

        super().__init__(opt)

@gin.configurable
class MuAdam(OptaxOptimizer):
    """Adam with a piecewise linear learning rate schedule."""

    def __init__(
        self,
        learning_rate,
        mup_lrs=None,
    ):
        assert mup_lrs is not None, "must provide mup_lrs"
        self.learning_rate = learning_rate

        def init_fn(params):
            del params
            return optax.EmptyState()
        
        def update_fn(updates, state, params=None):
            del params
            # print(updates)
            # print(mup_lrs)
            # import pdb; pdb.set_trace()
            updates = jax.tree_map(
                lambda update, scale: update * scale,
                updates,
                mup_lrs
            )
            # jax.tree_map(lambda update, scale: update * scale, updates, dict(mup_lrs))
            return updates, state
        
        opt = optax.chain(
            optax.adam(self.learning_rate),
            optax.GradientTransformation(init_fn, update_fn),
        )

        super().__init__(opt)

def _muadamw_schedule(args):

    def fix_dict(d):
        d = dict(d)
        for k,v in d.items():
            if isinstance(v, mmengine.config.config.ConfigDict):
                d[k] = fix_dict(v)

        return dict(d)
    # import pdb; pdb.set_trace()

    opt = MuAdam(**fix_dict(args.muadamw_schedule_kwargs))
    task = get_task(args)

    @jax.jit
    def update(opt_state, key, batch):
        params = opt.get_params(opt_state)

        if args.needs_state:
            state = opt.get_state(opt_state)
            (l, s), grad = jax.value_and_grad(task.loss_with_state, has_aux=True)(params, state, key, batch)
        else:
            l, grad = jax.value_and_grad(task.loss)(params, key, batch)
            s = None

        return opt.update(opt_state, grad, loss=l, model_state=s), l

    return opt, update

def _muadam(args):
    opt = MuAdam(args.learning_rate,args.runtime_mup_lrs)
    task = get_task(args)

    @jax.jit
    def update(opt_state, key, batch):
        params = opt.get_params(opt_state)

        if args.needs_state:
            state = opt.get_state(opt_state)
            (l, s), grad = jax.value_and_grad(task.loss_with_state, has_aux=True)(
                params, state, key, batch
            )
        else:
            l, grad = jax.value_and_grad(task.loss)(params, key, batch)
            s = None

        return opt.update(opt_state, grad, loss=l, model_state=s), l, grad

    return opt, update


@gin.configurable
class AdamW(OptaxOptimizer):
    """Adam with a piecewise linear learning rate schedule."""

    def __init__(
        self,
        learning_rate,
    ):
        opt = optax.adamw(learning_rate)
        super().__init__(opt)


def _sgd(args):
    opt = optax_opts.SGD(learning_rate=args.learning_rate)

    task = get_task(args)

    @jax.jit
    def update(opt_state, key, batch):
        params = opt.get_params(opt_state)

        if args.needs_state:
            state = opt.get_state(opt_state)
            (l, s), grad = jax.value_and_grad(task.loss_with_state, has_aux=True)(
                params, state, key, batch
            )
        else:
            l, grad = jax.value_and_grad(task.loss)(params, key, batch)
            s = None

        return opt.update(opt_state, grad, loss=l, model_state=s), l

    return opt, update


def _adam(args):
    opt = opt_base.Adam(args.learning_rate)

    task = get_task(args)

    @jax.jit
    def update(opt_state, key, batch):
        params = opt.get_params(opt_state)

        if args.needs_state:
            state = opt.get_state(opt_state)
            (l, s), grad = jax.value_and_grad(task.loss_with_state, has_aux=True)(
                params, state, key, batch
            )
        else:
            l, grad = jax.value_and_grad(task.loss)(params, key, batch)
            s = None

        return opt.update(opt_state, grad, loss=l, model_state=s), l

    return opt, update

def _AdamW(args):
    opt = AdamWLinearCosine(**args.adamw_schedule)

    task = get_task(args)

    @jax.jit
    def update(opt_state, key, batch):
        params = opt.get_params(opt_state)

        if args.needs_state:
            state = opt.get_state(opt_state)
            (l, s), grad = jax.value_and_grad(task.loss_with_state, has_aux=True)(
                params, state, key, batch
            )
        else:
            l, grad = jax.value_and_grad(task.loss)(params, key, batch)
            s = None

        return opt.update(opt_state, grad, loss=l, model_state=s), l, s

    return opt, update

def _fedlagg(args):
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

    with open(args.test_checkpoint, "rb") as f:
        meta_params = pickle.load(f)
    agg = lagg.opt_fn(meta_params)
    local_opt = optax_opts.SGD(learning_rate=args.local_learning_rate)
    task = get_task(args)

    def local_step(local_opt_state_and_key, local_batch):
        local_opt_state, key = local_opt_state_and_key
        params = local_opt.get_params(local_opt_state)
        key, key1 = jax.random.split(key)

        if args.needs_state:
            state = local_opt.get_state(local_opt_state)
            (l, s), grad = jax.value_and_grad(task.loss_with_state, has_aux=True)(params, state, key1, local_batch)
        else:
            print(type(task))
            l, grad = jax.value_and_grad(task.loss)(params, key1, local_batch)
            s = None

        return (local_opt.update(local_opt_state, grad, loss=l, model_state=s), key), l

    @functools.partial(jax.pmap, in_axes=(None, 0, 0), out_axes=(None, 0, None, None), axis_name="num_grads")
    def pmap_local_updates(init_local_opt_state, key, client_batch):
        (final_local_opt_state, _), local_losses = jax.lax.scan(local_step, (init_local_opt_state, key), client_batch)
        delta = jax.tree_util.tree_map(
            lambda new_p, old_p: new_p - old_p,
            local_opt.get_params(final_local_opt_state),
            local_opt.get_params(init_local_opt_state),
        )

        return (
            jax.lax.pmean(jnp.mean(local_losses), axis_name="num_grads"),
            delta,
            jax.lax.pmean(delta, axis_name="num_grads"),
            jax.lax.pmean(local_opt.get_state(final_local_opt_state), axis_name="num_grads") if args.needs_state else None
        )

    @functools.partial(jax.vmap, in_axes=(None, 0, 0))
    def vmap_local_updates_k(init_local_opt_state, key, client_batch):
        return jax.lax.scan(local_step, (init_local_opt_state, key), client_batch)
    
    devices = mesh_utils.create_device_mesh((globals.num_devices,1))
    mesh = Mesh(devices, ('i', 'j'))
    @functools.partial(jax.jit)
    @functools.partial(shard_map, 
                        mesh=mesh, 
                        in_specs=(P(),P('i',None),P('i',None),), 
                        out_specs=(P('i'),
                                    P('i'),
                                    P('i'),
                                    )
                        )
    def shard_map_local_updates(init_local_opt_state, key, client_batch):
        (final_local_opt_state, _), local_losses = vmap_local_updates_k(init_local_opt_state, key, client_batch)

        delta = jax.tree_util.tree_map(
            lambda new_p, old_p: new_p - old_p,
            local_opt.get_params(final_local_opt_state),
            local_opt.get_params(init_local_opt_state),
        )
        return (local_losses, delta, local_opt.get_state(final_local_opt_state) if globals.needs_state else None)

    @functools.partial(jax.vmap, in_axes=(None, 0, 0))
    def vmap_local_updates(init_local_opt_state, key, client_batch):
        (final_local_opt_state, _), local_losses = jax.lax.scan(local_step, (init_local_opt_state, key), client_batch)

        return (
            jnp.mean(local_losses),
            jax.tree_util.tree_map(
                lambda new_p, old_p: new_p - old_p,
                local_opt.get_params(final_local_opt_state),
                local_opt.get_params(init_local_opt_state),
            ),
            local_opt.get_state(final_local_opt_state) if args.needs_state else None,
        )

    def update(opt_state, key, batch):
        splitted_batches = jax.tree_util.tree_map(lambda x: x.reshape((args.num_grads, args.num_local_steps, args.local_batch_size) + x.shape[1:]), batch)
        init_local_opt_state = local_opt.init(agg.get_params(opt_state), model_state=agg.get_state(opt_state))

        keys = jax.random.split(key, args.num_grads)
        if args.use_pmap:
            losses, deltas, new_state = shard_map_local_updates(init_local_opt_state, keys, splitted_batches)

            loss = jnp.mean(losses)
            avg_delta = jax.tree_map(
                lambda ds: jnp.mean(ds, axis=0), deltas
            )
            if globals.needs_state:
                avg_state = jax.tree_map(lambda ns: jnp.mean(ns, axis=0),new_state)
            else:
                avg_state = None
        else:
            losses, deltas, new_state = vmap_local_updates(init_local_opt_state, keys, splitted_batches)
            loss = jnp.mean(losses)
            avg_delta = jax.tree_util.tree_map(
                    lambda ds: jnp.mean(ds, axis=0), deltas
            )
            if args.needs_state:
                avg_state = jax.tree_util.tree_map(
                    lambda s, ns: jnp.mean(ns, axis=0),
                    local_opt.get_state(init_local_opt_state),
                    new_state,
                )
            else:
                avg_state = None

        return agg.update(opt_state, deltas, avg_delta, loss=loss, model_state=avg_state), loss

    return agg, update


def _fedavg(args):
    local_opt = optax_opts.SGD(learning_rate=args.local_learning_rate)
    task = get_task(args)

    def local_step(local_opt_state_and_key, local_batch):
        local_opt_state, key = local_opt_state_and_key
        params = local_opt.get_params(local_opt_state)
        key, key1 = jax.random.split(key)

        if args.needs_state:
            state = local_opt.get_state(local_opt_state)
            (l, s), grad = jax.value_and_grad(task.loss_with_state, has_aux=True)(params, state, key1, local_batch)
        else:
            l, grad = jax.value_and_grad(task.loss)(params, key1, local_batch)
            s = None

        return (local_opt.update(local_opt_state, grad, loss=l, model_state=s), key), l

    @functools.partial(jax.pmap, in_axes=(None, 0, 0), out_axes=(None, None, None), axis_name="num_grads")
    def pmap_local_updates(init_local_opt_state, key, client_batch):
        (final_local_opt_state, _), local_losses = jax.lax.scan(local_step, (init_local_opt_state, key), client_batch)

        return (
            jax.lax.pmean(jnp.mean(local_losses), axis_name="num_grads"),
            jax.lax.pmean(
                local_opt.get_params(final_local_opt_state), axis_name="num_grads"
            ),
            jax.lax.pmean(
                local_opt.get_state(final_local_opt_state), axis_name="num_grads"
            ) if args.needs_state else None
        )

    @functools.partial(jax.vmap, in_axes=(None, 0, 0))
    def vmap_local_updates(init_local_opt_state, key, client_batch):
        (final_local_opt_state, _), local_losses = jax.lax.scan(local_step, (init_local_opt_state, key), client_batch)

        return (
            jnp.mean(local_losses),
            local_opt.get_params(final_local_opt_state),
            local_opt.get_state(final_local_opt_state) if args.needs_state else None,
        )

    def update(opt_state, key, batch):
        splitted_batches = jax.tree_util.tree_map(lambda x: x.reshape((args.num_grads, args.num_local_steps, args.local_batch_size) + x.shape[1:]), batch)

        keys = jax.random.split(key, args.num_grads)
        if args.use_pmap:
            loss, avg_params, avg_state = pmap_local_updates(opt_state, keys, splitted_batches)
        else:
            losses, new_params, new_state = vmap_local_updates(opt_state, keys, splitted_batches)
            loss = jnp.mean(losses)
            avg_params = jax.tree_util.tree_map(
                lambda p, nps: jnp.mean(nps, axis=0),
                local_opt.get_params(opt_state),
                new_params,
            )
            if args.needs_state:
                avg_state = jax.tree_util.tree_map(
                    lambda s, ns: jnp.mean(ns, axis=0),
                    local_opt.get_state(opt_state),
                    new_state,
                )
            else:
                avg_state = None

        return local_opt.init(avg_params, model_state=avg_state), loss

    return local_opt, update


def _fedavg_slowmo(args):
    opt = SGDSlowMo(learning_rate=args.local_learning_rate)
    task = get_task(args)

    def local_step(local_opt_state_and_key, local_batch):
        local_opt_state, key = local_opt_state_and_key
        params = opt.get_params(local_opt_state)
        key, key1 = jax.random.split(key)

        if args.needs_state:
            state = opt.get_state(local_opt_state)
            (l, s), grad = jax.value_and_grad(task.loss_with_state, has_aux=True)(params, state, key1, local_batch)
        else:
            l, grad = jax.value_and_grad(task.loss)(params, key1, local_batch)
            s = None

        return (opt.update(local_opt_state, grad, loss=l, model_state=s), key), l
    

    devices = mesh_utils.create_device_mesh((args.num_devices,1))
    mesh = Mesh(devices, ('i', 'j'))
    @functools.partial(jax.jit)
    @functools.partial(shard_map, 
                        mesh=mesh,
                        check_rep=False,
                        in_specs=(P(),P('i',None),P('i',None),),
                        out_specs=(P(None),
                                    P(None),
                                    P(None),)
                        )
    def shard_map_local_updates(init_local_opt_state, key, client_batch):
        key = jnp.squeeze(key)
        # print('init_local_opt_state',jax.tree_map(lambda x: x.shape, init_local_opt_state))
        # print('key',jax.tree_map(lambda x: x.shape, key))
        # print('client_batch',jax.tree_map(lambda x: x.shape, client_batch))
        client_batch = jax.tree_map(lambda x : jnp.squeeze(x, axis=0), client_batch)


        (final_local_opt_state, _), local_losses = jax.lax.scan(local_step, (init_local_opt_state, key), client_batch)

        return (
            jax.lax.pmean(jnp.mean(local_losses), axis_name="i"),
            jax.lax.pmean(
                opt.get_params(final_local_opt_state), axis_name="i"
            ),
            jax.lax.pmean(
                opt.get_state(final_local_opt_state), axis_name="i"
            ) if args.needs_state else None
        )


    @functools.partial(jax.pmap, in_axes=(None, 0, 0), out_axes=(None, None, None), axis_name="num_grads")
    def pmap_local_updates(init_local_opt_state, key, client_batch):
        (final_local_opt_state, _), local_losses = jax.lax.scan(local_step, (init_local_opt_state, key), client_batch)

        return (
            jax.lax.pmean(jnp.mean(local_losses), axis_name="num_grads"),
            jax.lax.pmean(
                opt.get_params(final_local_opt_state), axis_name="num_grads"
            ),
            jax.lax.pmean(
                opt.get_state(final_local_opt_state), axis_name="num_grads"
            ) if args.needs_state else None
        )

    @functools.partial(jax.vmap, in_axes=(None, 0, 0))
    def vmap_local_updates(init_local_opt_state, key, client_batch):
        (final_local_opt_state, _), local_losses = jax.lax.scan(local_step, (init_local_opt_state, key), client_batch)

        return (
            jnp.mean(local_losses),
            opt.get_params(final_local_opt_state),
            opt.get_state(final_local_opt_state) if args.needs_state else None,
        )

    def update(opt_state, key, batch):
        splitted_batches = jax.tree_util.tree_map(lambda x: x.reshape((args.num_grads, args.num_local_steps, args.local_batch_size) + x.shape[1:]), batch)

        keys = jax.random.split(key, args.num_grads)
        if args.use_pmap:
            loss, avg_params, avg_state = shard_map_local_updates(opt_state, keys, splitted_batches)
            # print('loss',jax.tree_map(lambda x: x.shape, loss))
            # print('avg_params',jax.tree_map(lambda x: x.shape, avg_params))
            # print('avg_state',jax.tree_map(lambda x: x.shape, avg_state))
            # exit(0)
        else:
            losses, new_params, new_state = vmap_local_updates(opt_state, keys, splitted_batches)
            loss = jnp.mean(losses)
            avg_params = jax.tree_util.tree_map(
                lambda p, nps: jnp.mean(nps, axis=0), opt.get_params(opt_state), new_params
            )
            if args.needs_state:
                avg_state = jax.tree_util.tree_map(
                    lambda s, ns: jnp.mean(ns, axis=0), opt.get_state(opt_state), new_state
                )
            else:
                avg_state = None

        ##### SLOW MO UPDATE (TODO not optimal) #####

        def update_momentum(momentum, avg_params, current_params, beta, local_learning_rate):
            return beta * momentum + (1 / local_learning_rate) * (current_params - avg_params)

        def update_params(current_params, momentum, local_learning_rate):
            return current_params - local_learning_rate * momentum

        # Get the momentum and current parameters
        momentum = opt_state.optax_opt_state[1]["momentum"]
        current_params = opt.get_params(opt_state)

        # Update the momentum
        momentum = jax.tree_util.tree_map(
            update_momentum,
            momentum,
            avg_params,
            current_params,
            jax.tree_util.tree_map(lambda x: args.beta, momentum),
            jax.tree_util.tree_map(lambda x: args.local_learning_rate, momentum),
        )

        # Update the parameters
        updated_params = jax.tree_util.tree_map(
            update_params,
            current_params,
            momentum,
            jax.tree_util.tree_map(lambda x: args.slowmo_learning_rate, current_params),
        )

        return opt.init(updated_params, momentum=momentum, model_state=avg_state), loss

    return opt, update


def _default_lopt(args):
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
                        mup_lrs=args.runtime_mup_lrs)
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
                        split_weights=False)

    with open(args.test_checkpoint, "rb") as f:
        meta_params = pickle.load(f)
    lopt = lopt.opt_fn(meta_params)
    task = get_task(args)

    def grad_step(carry, batch):
        params, state, key1, grad = carry


        print('batch in scal',jax.tree_map(lambda x: x.shape, batch))
        (nl, ns), ngrad = jax.value_and_grad(task.loss_with_state, has_aux=True)(params, state, key1, batch)
        # grad = jax.tree_map(jnp.add, ngrad, grad)
        # s = jax.tree_map(jnp.add, ns, s)
        # l = jax.tree_map(jnp.add, nl, l)
        new_accumulated_grads = jax.tree_map(jnp.add, grad, ngrad)

        return (params, state, key1, new_accumulated_grads),  (nl, ns)

    @functools.partial(jax.jit)
    def update(opt_state, key, batch):
        state = lopt.get_state(opt_state)
        params = lopt.get_params(opt_state)
        key, key1 = jax.random.split(key)
        gas = args.gradient_accumulation_steps


        # batch = jax.tree_map(lambda x:x.reshape((gas,x.shape[0]//gas,) + x.shape[1:]), batch)

        # for i in range(args.gradient_accumulation_steps):
        #     iterbatch = jax.tree_map(lambda x: x[i], batch)
        #     if i == 0:
        #         (l, s), grad = jax.value_and_grad(task.loss_with_state, has_aux=True)(params, state, key1, iterbatch)
        #     else:
        #         (nl, ns), ngrad = jax.value_and_grad(task.loss_with_state, has_aux=True)(params, state, key1, iterbatch)
        #         grad = jax.tree_map(jnp.add, ngrad, grad)
        #         s = jax.tree_map(jnp.add, ns, s)
        #         l = jax.tree_map(jnp.add, nl, l)
            
        # grad = jax.tree_map(lambda x: x / gas, grad)
        # s = jax.tree_map(lambda x: x / gas, s)
        # l = jax.tree_map(lambda x: x / gas, l)
            


        # print('before scan',jax.tree_map(lambda x: x.shape, batch))
        # grad = jax.tree_map(lambda x: jnp.zeros(x), params)
        # (_,_,_,grad),(l, s) = jax.lax.scan(grad_step,(params, state, key1, grad), batch)

        # print('grad',jax.tree_map(lambda x: x.shape, grad))
        # print('l',jax.tree_map(lambda x: x.shape, l))
        # print('s',jax.tree_map(lambda x: x.shape, s))
        # l = jax.tree_map(lambda x: jnp.mean(x,axis=0), l)
        # s = jax.tree_map(lambda x: jnp.mean(x,axis=0), s)

        # #compute the mean
        # grad = jax.tree_map(lambda x: x/gas, grad)

        # print('grad',jax.tree_map(lambda x: x.shape, grad))
        # print('l',jax.tree_map(lambda x: x.shape, l))
        # print('s',jax.tree_map(lambda x: x.shape, s))

        (l, s), grad = jax.value_and_grad(task.loss_with_state, has_aux=True)(params, state, key1, batch)


        
        return lopt.update(opt_state, grad, l, s), l, grad

    return lopt, update


def _velo(args):
    lopt = prefab.LearnedOptimizer(args.num_inner_steps)

    task = get_task(args)

    @jax.jit
    def update(opt_state, key, batch):
        params = lopt.get_params(opt_state)

        if args.needs_state:
            state = lopt.get_state(opt_state)
            (l, s), grad = jax.value_and_grad(task.loss_with_state, has_aux=True)(
                params, state, key, batch
            )
        else:
            l, grad = jax.value_and_grad(task.loss)(params, key, batch)
            s = None

        return lopt.update(opt_state, grad, loss=l, model_state=s), l, s

    return lopt, update


def get_optimizer(args):
    optimizers = {
        "adamw": _AdamW,
        "adam": _adam,
        "sgd": _sgd,
        "fedavg": _fedavg,
        "fedavg-slowmo": _fedavg_slowmo,
        "fedlopt": _fedlagg,
        "fedlopt-adafac": _fedlagg,
        "fedlagg": _fedlagg,
        "fedlagg-wavg": _fedlagg,
        "fedlagg-adafac": _fedlagg,
        "small_fc_mlp":_default_lopt,
        "mup_small_fc_mlp":_default_lopt,
        "velo": _velo,
        'muadam':_muadam,
    }

    return optimizers[args.optimizer](args)  # TODO Find better way to do this
