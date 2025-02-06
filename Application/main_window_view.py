from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QDoubleValidator
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from datetime import datetime

from main_window_ui import Ui_mainWindow
from equation_constant import EquationConstant
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import platform
if platform.system() == "Darwin":
    matplotlib.use("Qt5Agg")

import os.path
from PyQt5.QtWidgets import QApplication

class MainWindowView(QtWidgets.QMainWindow, Ui_mainWindow):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setupUi(self)
        self.window().setWindowTitle("Glioma Growth Visualization")

        # Default Setting
        self.flair_rb.setChecked(True)   # Flair is selected
        self.toggle_checkbox.setChecked(True)  # toggle is selected
        self.process_info_label.hide()
        self.disable_by_start(False)
        self.time_slider.setMaximum(EquationConstant.NUM_STEPS)
        self.set_input_range_label()

        # Restrict user input
        self.diffusion_rate_input.setValidator(QDoubleValidator(EquationConstant.MIN_DIFFUSION, EquationConstant.MAX_DIFFUSION, 1))
        self.reaction_rate_input.setValidator(QDoubleValidator(EquationConstant.MIN_REACTION, EquationConstant.MAX_REACTION, 2))
        # Set user input to default value
        self.set_default_input()
        # self.diffusion_rate_input.setText(str(EquationConstant.DIFFUSION_RATE))
        # self.reaction_rate_input.setText(str(EquationConstant.REACTION_RATE))

        # Default resize
        self.process_info_label.setMaximumHeight(20)  # Resize processing information label
        self.equation_widget.setMinimumHeight(300)  # Resize equation widget size

        # Assign user actions
        self.actionSave.triggered.connect(self.save_screen)

        self.flair_file_button.clicked.connect(lambda: self.selected_file_clicked(EquationConstant.FLAIR_KEY))
        self.glistrboost_file_button.clicked.connect(
            lambda: self.selected_file_clicked(EquationConstant.GLISTRBOOST_KEY))
        self.t1_file_button.clicked.connect(lambda: self.selected_file_clicked(EquationConstant.T1_KEY))
        self.t1gd_file_button.clicked.connect(lambda: self.selected_file_clicked(EquationConstant.T1GD_KEY))
        self.t2_file_button.clicked.connect(lambda: self.selected_file_clicked(EquationConstant.T2_KEY))

        self.flair_rb.toggled.connect(self.update_plt)
        self.t1_rb.toggled.connect(self.update_plt)
        self.t1gd_rb.toggled.connect(self.update_plt)
        self.t2_rb.toggled.connect(self.update_plt)

        self.toggle_checkbox.clicked.connect(self.update_plt)

        self.slice_slider.sliderReleased.connect(self.update_plt)
        self.time_slider.sliderReleased.connect(self.update_plt)

        self.start_button.clicked.connect(self.start_equation)
        self.reset_button.clicked.connect(self.reset_equation)

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
        if self.check_files():
            self.process_info_label.show()
            QApplication.processEvents()
            diffusion = self.get_diffusion()
            reaction = self.get_reaction()
            self.controller.run_equation_model(diffusion, reaction, self.get_cur_scan())
            self.disable_by_start(True)
            self.equation_running_info_label.setText(f"Running Equation Model with diffusion rate {diffusion} and reaction rate {reaction}")

    def check_files(self):
        # TODO: Simple check, need to be updated!
        returned_val = False
        msg = ""
        if  self.flair_file_label.text() == "":
            msg = "Missing FLAIR File"
        elif self.glistrbosst_file_label.text() == "":
            msg = "Missing GLISTROBOSST File"
        elif self.t1_file_label.text() == "":
            msg = "Missing T1 File"
        elif self.t1gd_file_label.text() == "":
            msg = "Missing T1GD File"
        elif self.t2_file_label.text() == "":
            msg = "Missing T2 File"
        else:
            returned_val = True
        if not returned_val:
            message_box = QMessageBox()
            message_box.setText(msg)
            message_box.setIcon(QMessageBox.Information)
            message_box.setStandardButtons(QMessageBox.Ok)
            message_box.exec_()
        return returned_val

    def reset_equation(self):
        self.disable_by_start(False)
        self.set_input_range_label()
        self.set_default_input()

    def set_default_input(self):
        self.diffusion_rate_input.setText(str(EquationConstant.DIFFUSION_RATE))
        self.reaction_rate_input.setText(str(EquationConstant.REACTION_RATE))

    def set_input_range_label(self):
        self.equation_running_info_label.setText(f"Diffusion Rate Range: [{EquationConstant.MIN_DIFFUSION},{EquationConstant.MAX_DIFFUSION}], "
                                                 f"Reaction Rate Range: [{EquationConstant.MIN_REACTION}，{EquationConstant.MAX_REACTION}]")

    # def (self):
    #     value = self.slice_slider.value()
    #     print(f"slider released {value}")

    def init_sliders(self, cur_slice, max_slice):
        self.slice_slider.setSliderPosition(cur_slice)
        self.slice_slider.setMaximum(max_slice)
        self.time_slider.setSliderPosition(0)
        self.update_slider_value_labels(0)

    def update_slider_value_labels(self, time_val):
        self.slice_value_label.setText(str(self.slice_slider.value()))
        self.time_value_label.setText(f"{time_val} days")
        self.process_info_label.hide()

    def update_plt(self):
        self.process_info_label.show()
        QApplication.processEvents()
        scan = self.get_cur_scan()
        slice_i = self.slice_slider.value()
        time_i = self.time_slider.value()
        is_overlay = self.toggle_checkbox.isChecked()
        self.controller.process_plts(scan, slice_i, time_i, is_overlay)

    def get_cur_scan(self):
        if self.t1_rb.isChecked():
            scan = 't1'
        elif self.t2_rb.isChecked():
            scan = 't2'
        elif self.t1gd_rb.isChecked():
            scan = 't1gd'
        else:
            scan = 'flair'
        return scan


    def disable_by_start(self, has_start):
        self.start_button.setDisabled(has_start)
        self.reset_button.setDisabled(not has_start)
        self.disable_input_lineedit(has_start)
        self.disable_file_selection(has_start)
        self.disable_radio_buttons(not has_start)
        self.toggle_checkbox.setDisabled(not has_start)
        self.disable_sliders(not has_start)

    def disable_input_lineedit(self, disable):
        self.diffusion_rate_input.setDisabled(disable)
        self.reaction_rate_input.setDisabled(disable)

    def disable_file_selection(self, disable):
        self.flair_file_button.setDisabled(disable)
        self.glistrboost_file_button.setDisabled(disable)
        self.t1_file_button.setDisabled(disable)
        self.t1gd_file_button.setDisabled(disable)
        self.t2_file_button.setDisabled(disable)

    def disable_radio_buttons(self, disable):
        self.flair_rb.setDisabled(disable)
        self.t1_rb.setDisabled(disable)
        self.t1gd_rb.setDisabled(disable)
        self.t2_rb.setDisabled(disable)

    def disable_sliders(self, disable):
        self.time_slider.setDisabled(disable)
        self.slice_slider.setDisabled(disable)

    def update_equation_graph(self, fig):
        # self.canvas = FigureCanvasQTAgg(fig)
        # self.toolbar = NavigationToolbar2QT(self.canvas, self)
        # self.equation_layout.addWidget(self.canvas)
        if self.equation_layout.count() > 0:
            self.equation_layout.removeWidget(self.equation_layout.itemAt(0).widget())
        self.equation_layout.addWidget(FigureCanvasQTAgg(fig))
        self.process_info_label.hide()

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

    def save_screen(self):
        screenshot = self.window().grab()
        saved_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        screenshot.save(saved_name + ".png", "PNG")
        self.save_popup()

    def save_popup(self):
        message_box = QMessageBox()
        message_box.setText("Screenshot Saved!")
        message_box.setIcon(QMessageBox.Information)
        message_box.setStandardButtons(QMessageBox.Ok)
        message_box.exec_()