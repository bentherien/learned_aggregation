_base_ = ["./schedule_base.py"]
adamw_schedule = dict(
    init_value=0,
    peak_value=3e-3,
    end_value=3e-4,
    warmup_steps=10,
    decay_steps=990,
    exponent=1.0,
    clip=True,
)