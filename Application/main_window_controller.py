import subprocess
import sys
import os
import threading
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # for importing from biological_model

from biological_model import BiologicalModel
from Application.main_window_view import MainWindowView
from PyQt5.QtCore import QRunnable, pyqtSignal, QObject, QThread



class MainWindowController():
    _instance = None
    # update_ui = pyqtSignal(list)

    def __init__(self):
        # super(MainWindowController, self).__init__()

        self.equation_model = BiologicalModel.instance()
        self.view = MainWindowView(self)
        self.diffusion = None
        self.reaction = None
        self.scan = None
        self.has_start = False

    #
    # def print_thread_functions(self):
    #     print(f"Total Threads: {len(threading.enumerate())}")
    #     for thread in threading.enumerate():
    #         print(f"\nThread: {thread.name}")
    #         frame = sys._current_frames().get(thread.ident, None)
    #         if frame:
    #             stack = traceback.format_stack(frame)
    #             print("".join(stack))

    # def run(self):
    #     print("run run")
    #
    #     self.print_thread_functions()
    #     self.run_equation_model(self.diffusion, self.reaction, self.scan)
    #     self.print_thread_functions()

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = MainWindowController()
        return cls._instance

    def set_before_run(self, diffusion, reaction, scan):
        self.diffusion = diffusion
        self.reaction = reaction
        self.scan = scan

    def run_equation_model(self, diffusion, reaction, scan):

        self.equation_model.set_diffusion_rate(diffusion)
        self.equation_model.set_reaction_rate(reaction)
        resultFig, cur_slice_index, max_slices = self.equation_model.start_equation(scan)
        self.view.init_sliders(cur_slice_index, max_slices)
        self.view.update_equation_graph(resultFig)
        # self.update_ui.emit(resultFig)

    def set_selected_file(self, file_key, file_path):
        self.equation_model.update_file_paths(file_key, file_path)

    def process_plts(self, scan, slice_i, time_i, is_overlay):
        self.equation_model.update(slice_i, time_i, is_overlay, scan)
        time_day = int(self.equation_model.time_in_days(time_i))
        self.view.update_slider_value_labels(time_day)