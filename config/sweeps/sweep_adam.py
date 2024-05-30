_base_ = ["./sweeps_base.py"]

optimizer = "adam"
task = "mlp128x128x128_imagenet_32"
num_inner_steps = 1000
num_grads = 8
num_local_steps = 4

sweep_config = dict(
    method="grid",
    metric=dict(name="test loss", goal="minimize"),
    parameters=dict(
        learning_rate=dict(
            values=[
                1,
                0.5,
                0.1,
                0.05,
                0.01,
                0.005,
                0.001,
                0.0005,
                0.0001,
            ]
        ),
    ),
)
