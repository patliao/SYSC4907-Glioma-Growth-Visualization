

class BiologicalInfo:
    _instance = None

    def __init__(self):
        self.file_path = None
        self.diffusion_mask = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = BiologicalInfo()
        return cls._instance

    def __setattr__(self, key, value):
        if key == 'diffusion_mask' and value is not None:
            print("successfully set diffusion mask")
        super(BiologicalInfo, self).__setattr__(key, value)