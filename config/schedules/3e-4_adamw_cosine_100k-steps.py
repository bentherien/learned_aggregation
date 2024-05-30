_base_ = ["./schedule_base.py"]
adamw_schedule = dict(
    init_value=0,
    peak_value=3e-4,
    end_value=3e-5,
    warmup_steps=50,
    decay_steps=99950,
    exponent=1.0,
    clip=True,
)