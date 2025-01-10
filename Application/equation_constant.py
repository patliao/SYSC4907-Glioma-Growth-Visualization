class EquationConstant:

    FILE_KEYS = ['flair', 'glistrboost', 't1', 't1gd', 't2']
    FLAIR_KEY = 'flair'
    GLISTRBOOST_KEY = 'glistrboost'
    T1_KEY = 't1'
    T1GD_KEY = 't1gd'
    T2_KEY = 't2'

    SPATIAL_RESOLUTION = 1.0  # mm
    DIFFUSION_RATE = 0.5  # mm/day
    REACTION_RATE = 0.01  # per day
    NUM_STEPS = 500  # number of steps in time in the model

    MIN_DIFFUSION = 0.1
    MAX_DIFFUSION = 1.5
    MIN_REACTION = 0.01
    MAX_REACTION = 1.0
