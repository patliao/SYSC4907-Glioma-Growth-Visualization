from Application.Controller.file_selection_controller import FileSelectionController
from Application.Controller.setting_controller import SettingController
from Application.View.main_window_view import MainWindowView


class MainWindowController:
    _instance = None

    def __init__(self):
        self.view = MainWindowView(self)
        self.view.initialize_widgets(FileSelectionController.instance().view, SettingController.instance().view, None)

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = MainWindowController()
        return cls._instance
