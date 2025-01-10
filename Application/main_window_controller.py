
from biological_model import BiologicalModel
from Application.main_window_view import MainWindowView


class MainWindowController:
    _instance = None

    def __init__(self):
        self.equation_model = BiologicalModel.instance()
        self.view = MainWindowView(self)

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = MainWindowController()
        return cls._instance

    def run_equation_model(self, diffusion, reaction):
        self.equation_model.set_diffusion_rate(diffusion)
        self.equation_model.set_reaction_rate(reaction)
        resultFig = self.equation_model.start_equation()
        self.view.update_equation_graph(resultFig)

    def set_selected_file(self, file_key, file_path):
        self.equation_model.update_file_paths(file_key, file_path)