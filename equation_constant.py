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
    NUM_STEPS = 500  # number of steps in time in the model

    WHITE_DIFFUSION_RATE = 0.088 # Dw > 5Dg (mm^2/day)
    GREY_DIFFUSION_RATE = WHITE_DIFFUSION_RATE/5 # mm^2/day
    CSF_DIFFUSION_RATE = 0.01 # ~0 (mm^2/day)

    DIFFUSION_RATE = WHITE_DIFFUSION_RATE # mm²/day
    REACTION_RATE = 0.029  # per day

    LAMBDA = np.sqrt(WHITE_DIFFUSION_RATE / REACTION_RATE) # infiltration length of glioma cells in white matter (mm)

    MIN_DIFFUSION = 0.00001
    MAX_DIFFUSION = 1.5
    MIN_REACTION = 0.00001
    MAX_REACTION = 1.0

    MAX_DIFFUSION = max(
            CSF_DIFFUSION_RATE,
            GREY_DIFFUSION_RATE,
            WHITE_DIFFUSION_RATE
        )
    TIME_STEP = (SPATIAL_RESOLUTION ** 2) / (2 * 3 * MAX_DIFFUSION)

    SAG = "sag"
    COR = "cor"
    AXI = "axi"

