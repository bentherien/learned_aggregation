_base_ = ["./sweeps_base.py"]

optimizer = "fedavg-slowmo"
task = "mlp128x128x128_imagenet_32"
num_inner_steps = 1000
num_grads = 8
num_local_steps = 4

sweep_config = dict(
    method="grid",
    metric=dict(name="outer valid loss", goal="minimize"),
    parameters=dict(
        slowmo_learning_rate=dict(
            values=[
                0.5,
                0.1,
                0.05,
                0.01,
                0.005,
                0.001,
            ]
        ),
        local_learning_rate=dict(
            values=[
                1.0,
                0.5,
                0.3,
                0.1,
            ]
        ),
        beta=dict(values=[0.99, 0.95, 0.9, 0.85, 0.8]),
    ),
)


