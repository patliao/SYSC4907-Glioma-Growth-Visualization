class EquationConstant:

    FILE_KEYS = ['flair', 'glistrboost', 't1', 't1gd', 't2']
    FLAIR_KEY = 'flair'
    GLISTRBOOST_KEY = 'glistrboost'
    T1_KEY = 't1'
    T1GD_KEY = 't1gd'
    T2_KEY = 't2'

    SPATIAL_RESOLUTION = 1.0  # mm
    DIFFUSION_RATE = 0.236 # mm²/day
    REACTION_RATE = 0.012  # per day
    NUM_STEPS = 500  # number of steps in time in the model

    GREY_DIFFUSION_RATE = 0.0393 # mm^2/day
    WHITE_DIFFUSION_RATE = 0.1967 # Dw = 5Dg (mm^2/day)
    CSF_DIFFUSION_RATE = 0.5901 # Dcsf = 3Dw (mm^2/day)

    MIN_DIFFUSION = 0.1
    MAX_DIFFUSION = 1.5
    MIN_REACTION = 0.01
    MAX_REACTION = 1.0