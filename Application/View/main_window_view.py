from PyQt5 import QtWidgets

from Application.UI_Code.main_window_ui import Ui_mainWindow


class MainWindowView(QtWidgets.QMainWindow, Ui_mainWindow):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setupUi(self)
        self.show()

    def initialize_widgets(self, file_selection, setting, equation):
        self.input_and_setting.layout().addWidget(file_selection)
        self.input_and_setting.layout().addWidget(setting)
        if equation is not None:
            self.equation_frame.layout().addWidget(equation)

