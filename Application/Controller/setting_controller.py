from Application.Model.biological_model import BiologicalModel
from Application.View.setting_view import SettingView


class SettingController:
    _instance = None

    def __init__(self):
        self.view = SettingView(self)
        self.equation_model = BiologicalModel.instance()

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = SettingController()
        return cls._instance

    def run_equation_model(self):
        self.equation_model.start_equation()
