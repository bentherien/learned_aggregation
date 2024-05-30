_base_ = ["./sweeps_base.py"]


num_inner_steps = 1000

sweep_config = dict(
    method="grid",
    metric=dict(name="train loss", goal="minimize"),
    parameters=dict(
        learning_rate=dict(
            values=(
                0.1,
                0.01,
                0.001,
                0.0001,
            )
        ),
        mup_input_mult=dict(
            values=[
                2 ** (4),
                2 ** (2),
                1.0,
                2 ** (-2),
                2 ** (-4),
            ]
        ),

        mup_output_mult=dict(
            values=[
                2 ** (4),
                2 ** (2),
                1.0,
                2 ** (-2),
                2 ** (-4),
            ]
        ),

        mup_hidden_lr_mult=dict(
            values=[
                2 ** (4),
                2 ** (2),
                1.0,
                2 ** (-2),
                2 ** (-4),
            ]
        ),
    ),
)
