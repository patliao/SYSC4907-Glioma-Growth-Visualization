import numpy as np
class EquationConstant:

    FILE_KEYS = ['flair', 'glistrboost', 't1', 't1gd', 't2', 'seg2']
    FLAIR_KEY = 'flair'
    GLISTRBOOST_KEY = 'glistrboost'
    T1_KEY = 't1'
    T1GD_KEY = 't1gd'
    T2_KEY = 't2'
    SEG2_KEY = 'seg2'

    SPATIAL_RESOLUTION = 1.0  # mm
    NUM_STEPS = 600  # number of steps in time in the model

    GREY_DIFFUSION_RATE = 0.13 # mm^2/day
    WHITE_DIFFUSION_RATE = 0.65 # Dw = 5Dg (mm^2/day)
    CSF_DIFFUSION_RATE = 1.95 # Dcsf = 3Dw (mm^2/day)

    DIFFUSION_RATE = WHITE_DIFFUSION_RATE # mm²/day
    REACTION_RATE = 0.012  # per day

    LAMBDA = np.sqrt(WHITE_DIFFUSION_RATE / REACTION_RATE) # infiltration length of glioma cells in white matter (mm)

    MIN_DIFFUSION = 0.1
    MAX_DIFFUSION = 1.5
    MIN_REACTION = 0.01
    MAX_REACTION = 1.0

    MAX_DIFFUSION = max(
            CSF_DIFFUSION_RATE,
            GREY_DIFFUSION_RATE,
            WHITE_DIFFUSION_RATE
        )
    TIME_STEP = (SPATIAL_RESOLUTION ** 2) / (2 * 3 * MAX_DIFFUSION)