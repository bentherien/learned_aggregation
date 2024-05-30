_base_ = ["../meta_train_base.py"]
schedule = dict(
    init_value=3e-10,
    peak_value=3e-2,
    end_value=(3e-2) * 1/3,
    warmup_steps=100,
    decay_steps=4900,
    exponent=1.0,
    clip=True,
)

num_outer_steps = 5000
num_inner_steps = 1000