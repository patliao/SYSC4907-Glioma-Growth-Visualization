from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QDoubleValidator
from matplotlib.backends.backend_qt import NavigationToolbar2QT

from Application.main_window_ui import Ui_mainWindow
from Application.equation_constant import EquationConstant
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import os.path

class MainWindowView(QtWidgets.QMainWindow, Ui_mainWindow):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setupUi(self)

        # Restrict user input
        self.diffusion_rate_input.setValidator(QDoubleValidator(EquationConstant.MIN_DIFFUSION, EquationConstant.MAX_DIFFUSION, 1))
        self.reaction_rate_input.setValidator(QDoubleValidator(EquationConstant.MIN_REACTION, EquationConstant.MAX_REACTION, 2))
        # Set user input to default value
        self.diffusion_rate_input.setText(str(EquationConstant.DIFFUSION_RATE))
        self.reaction_rate_input.setText(str(EquationConstant.REACTION_RATE))

        # Resize processing information label
        self.process_info_label.setMaximumHeight(5)
        # Resize equation widget size
        self.equation_widget.setMinimumHeight(500)

        # Assign user actions
        self.flair_file_button.clicked.connect(lambda: self.selected_file_clicked(EquationConstant.FLAIR_KEY))
        self.glistrboost_file_button.clicked.connect(
            lambda: self.selected_file_clicked(EquationConstant.GLISTRBOOST_KEY))
        self.t1_file_button.clicked.connect(lambda: self.selected_file_clicked(EquationConstant.T1_KEY))
        self.t1gd_file_button.clicked.connect(lambda: self.selected_file_clicked(EquationConstant.T1GD_KEY))
        self.t2_file_button.clicked.connect(lambda: self.selected_file_clicked(EquationConstant.T2_KEY))

        self.start_button.clicked.connect(self.start_equation)
        self.reset_button.clicked.connect(self.reset_equation)

        # Default Setting
        self.flair_rb.setChecked(True)   # Flair is selected
        self.toggle_checkbox.setChecked(True)  # toggle is selected

        self.auto_selection()

        self.show()

    def update_equation_default(self):
        self.diffusion_rate_input.setText(str(EquationConstant.DIFFUSION_RATE))
        self.reaction_rate_input.setText(str(EquationConstant.REACTION_RATE))

    def selected_file_clicked(self, button_key):
        file_name, file_path = self.file_select_dialog()
        self.update_selected_file_info(button_key, file_path, file_name)

    def update_selected_file_info(self, button_key, file_path, file_name):
        self.controller.set_selected_file(button_key, file_path)
        self.update_selected_file_label(button_key, file_name)

    def update_selected_file_label(self, label_key, file_name):
        if label_key == EquationConstant.FLAIR_KEY:
            self.flair_file_label.setText(file_name)
        elif label_key == EquationConstant.GLISTRBOOST_KEY:
            self.glistrbosst_file_label.setText(file_name)
        elif label_key == EquationConstant.T1_KEY:
            self.t1_file_label.setText(file_name)
        elif label_key == EquationConstant.T1GD_KEY:
            self.t1gd_file_label.setText(file_name)
        else:
            self.t2_file_label.setText(file_name)

    def file_select_dialog(self):
        dlg = QFileDialog()
        filePath, _ = dlg.getOpenFileName(self, "Open File", "", "All Files (*);;Text Files (*.nii)")
        fileName = filePath.split("/")[-1]
        # print("selected file path: " + str(filePath))
        # print("selected file name: " + str(fileName))
        return fileName, filePath

    def start_equation(self):
        diffusion = self.get_diffusion()
        reaction = self.get_reaction()
        self.controller.run_equation_model(diffusion, reaction)
        self.start_button.setDisabled(True)
        self.reset_button.setDisabled(False)
        self.disable_input_lineedit(True)
        self.equation_running_info_label.setText(f"Running Equation Model with diffusion rate {diffusion} and reaction rate {reaction}")

    def reset_equation(self):
        self.start_button.setDisabled(False)
        self.reset_button.setDisabled(True)
        self.disable_input_lineedit(False)
        self.equation_running_info_label.setText(f"Diffusion Rate Range: [{EquationConstant.MIN_DIFFUSION},{EquationConstant.MAX_DIFFUSION}], "
                                                 f"Reaction Rate Range: [{EquationConstant.MIN_REACTION}，{EquationConstant.MAX_REACTION}]")

    def disable_input_lineedit(self, disable):
        self.diffusion_rate_input.setDisabled(disable)
        self.reaction_rate_input.setDisabled(disable)

    def update_equation_graph(self, fig):
        # self.canvas = FigureCanvasQTAgg(fig)
        # self.toolbar = NavigationToolbar2QT(self.canvas, self)
        # self.equation_layout.addWidget(self.canvas)
        self.equation_layout.addWidget(FigureCanvasQTAgg(fig))

    def auto_selection(self):
        """
        Selects specific MRI files automatically. Use for testing
        """
        try:
            current_file_path = os.path.dirname(__file__)
            # print("current path: " + str(current_file_path))
            parent_path = os.path.split(current_file_path)[0]
            # print("parent path: " + str(parent_path))
            testing_files_path = os.path.join(parent_path, "TCGA-HT-8111")
            # print("testing path: " + str(testing_files_path))
            for filename in os.listdir(testing_files_path):
                file_path = os.path.join(testing_files_path, filename)
                if filename.__contains__("flair.nii"):
                    self.update_selected_file_info(EquationConstant.FLAIR_KEY, file_path, filename)
                elif filename.__contains__("GlistrBoost.nii"):
                    self.update_selected_file_info(EquationConstant.GLISTRBOOST_KEY, file_path, filename)
                elif filename.__contains__("t1.nii"):
                    self.update_selected_file_info(EquationConstant.T1_KEY, file_path, filename)
                elif filename.__contains__("t1Gd.nii"):
                    self.update_selected_file_info(EquationConstant.T1GD_KEY, file_path, filename)
                elif filename.__contains__("t2.nii"):
                    self.update_selected_file_info(EquationConstant.T2_KEY, file_path, filename)
        except:
            print("Auto Selection Fail!")

    def get_diffusion(self):
        diffusion_rate =EquationConstant.DIFFUSION_RATE
        try:
            diffusion_rate = float(self.diffusion_rate_input.text())
            if diffusion_rate < EquationConstant.MIN_DIFFUSION:
                diffusion_rate = EquationConstant.MIN_DIFFUSION
            elif diffusion_rate > EquationConstant.MAX_DIFFUSION:
                diffusion_rate = EquationConstant.MAX_DIFFUSION
        except:
            pass
        self.diffusion_rate_input.setText(str(diffusion_rate))
        return diffusion_rate

    def get_reaction(self):
        reaction_rate =EquationConstant.REACTION_RATE
        try:
            reaction_rate = float(self.reaction_rate_input.text())
            if reaction_rate < EquationConstant.MIN_REACTION:
                reaction_rate = EquationConstant.MIN_REACTION
            elif reaction_rate > EquationConstant.MAX_REACTION:
                reaction_rate = EquationConstant.MAX_REACTION
        except:
            pass
        self.reaction_rate_input.setText(str(reaction_rate))
        return reaction_rate