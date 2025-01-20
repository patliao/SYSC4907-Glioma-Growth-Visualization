
from biological_model import BiologicalModel
from Application.main_window_view import MainWindowView
from PyQt5.QtCore import QThread


class MainWindowController:
    _instance = None

    def __init__(self):
        # super().__init__()
        self.equation_model = BiologicalModel.instance()
        self.view = MainWindowView(self)
        # self.start()  # Start the thread

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = MainWindowController()
        return cls._instance

    def run_equation_model(self, diffusion, reaction, scan):
        self.equation_model.set_diffusion_rate(diffusion)
        self.equation_model.set_reaction_rate(reaction)
        resultFig, cur_slice_index, max_slices = self.equation_model.start_equation(scan)
        self.view.init_sliders(cur_slice_index, max_slices)
        self.view.update_equation_graph(resultFig)

    def set_selected_file(self, file_key, file_path):
        self.equation_model.update_file_paths(file_key, file_path)

    def process_plts(self, scan, slice_i, time_i, is_overlay):
        self.equation_model.update(slice_i, time_i, is_overlay, scan)
        time_day = int(self.equation_model.time_in_days(time_i))
        self.view.update_slider_value_labels(time_day)