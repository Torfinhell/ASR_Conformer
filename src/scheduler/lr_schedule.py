def lr_by_step( warmup_steps, model_dim):
    peak_lr=00.5/(model_dim**(-0.5))
    return lambda step:(
        (step / warmup_steps)**(0.5) *peak_lr
    if step < warmup_steps
    else min(step**(-0.5), step * warmup_steps**(-1.5)) *  peak_lr
    )