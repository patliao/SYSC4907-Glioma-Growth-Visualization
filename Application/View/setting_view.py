from PyQt5 import QtWidgets

from Application.UI_Code.setting_ui import Ui_setting_widget


class SettingView(QtWidgets.QWidget, Ui_setting_widget):
    def __init__(self, controller):
        super().__init__()
        self.setupUi(self)
        self.controller = controller

        self.start_button.clicked.connect(self.run_equation)

        self.show()

    def run_equation(self):
        self.controller.run_equation_model()