
from PyQt5 import QtWidgets, Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap, QColor
from datetime import datetime
from PyQt5.QtCore import Qt

from typing_extensions import override

from newMainWindow import Ui_mainWindow
from equation_constant import EquationConstant
import matplotlib
import platform

from separateComparison import SeparateComparison

if platform.system() == "Darwin":
    matplotlib.use("Qt5Agg")

import os.path
from PyQt5.QtWidgets import QApplication
from pyqtspinner import WaitingSpinner
from main_window_controller import MainWindowController

class MainWindowView(QtWidgets.QMainWindow, Ui_mainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # self.setFixedSize(1800, 1200)
        self.controller = MainWindowController.instance()
        self.setupUi(self)
        self.window().setWindowTitle("Glioma Growth Visualization")

        # Default Setting
        self.flair_rb.setChecked(True)   # Flair is selected
        self.toggle_checkbox.setChecked(True)  # toggle is selected
        self.process_info_label.hide()
        self.disable_by_start(False)
        self.time_slider.setMaximum(EquationConstant.NUM_STEPS)
        # self.set_input_range_label()

        # Set user input to default value
        self.set_default_input()

        # Default resize
        self.process_info_label.setMaximumHeight(20)  # Resize processing information label
        self.equation_widget.setMinimumHeight(300)  # Resize equation widget size

        # Assign user actions
        self.actionSave.triggered.connect(self.save_screen)
        self.actionSave_Mask.triggered.connect(self.save_mask)

        self.flair_file_button.clicked.connect(lambda: self.selected_file_clicked(EquationConstant.FLAIR_KEY))
        self.glistrboost_file_button.clicked.connect(
            lambda: self.selected_file_clicked(EquationConstant.GLISTRBOOST_KEY))
        self.t1_file_button.clicked.connect(lambda: self.selected_file_clicked(EquationConstant.T1_KEY))
        self.t1gd_file_button.clicked.connect(lambda: self.selected_file_clicked(EquationConstant.T1GD_KEY))
        self.t2_file_button.clicked.connect(lambda: self.selected_file_clicked(EquationConstant.T2_KEY))
        self.seg_file_button.clicked.connect(lambda: self.selected_file_clicked(EquationConstant.SEG2_KEY))
        self.flair2_file_button.clicked.connect(lambda: self.selected_file_clicked("flair 2"))

        self.flair_rb.toggled.connect(self.update_plt)
        self.t1_rb.toggled.connect(self.update_plt)
        self.t1gd_rb.toggled.connect(self.update_plt)
        self.t2_rb.toggled.connect(self.update_plt)

        self.equation_checkBox.toggled.connect(self.update_image_display)
        self.mix_checkBox.toggled.connect(self.update_image_display)
        self.real_checkBox.toggled.connect(self.update_image_display)
        self.ai_checkBox.toggled.connect(self.update_image_display)

        self.toggle_checkbox.clicked.connect(self.update_plt)

        self.slice_slider.sliderMoved.connect(self.update_plt)
        self.time_slider.sliderMoved.connect(self.update_plt)

        self.start_button.clicked.connect(self.start_equation)
        self.reset_button.clicked.connect(self.reset_equation)
        self.separate_button.clicked.connect(self.popup_detail)

        self.sagittal_image_label.setScaledContents(True)
        self.coronal_label_image.setScaledContents(True)
        self.axial_label_image.setScaledContents(True)

        self.spinner = self.createSpinner()
        self.auto_selection()

        self.controller.initSliders.connect(self.init_sliders)
        self.controller.updatePlot.connect(self.update_plot)
        self.controller.updateTime.connect(self.update_slider_value_labels)
        self.controller.stopSpinner.connect(self.stop_spinner)

        # self.sagittal_image_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        # self.sagittal_image_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        # self.sagittal_image_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.sagittal_image_label.setFixedHeight(160)
        self.sagittal_image_label.setFixedWidth(240)
        self.coronal_label_image.setFixedHeight(160)
        self.coronal_label_image.setFixedWidth(260)
        self.axial_label_image.setFixedHeight(240)
        self.axial_label_image.setFixedWidth(240)

        # self.sagittal_image_label.setMinimumSize(160, 240)
        # self.coronal_label_image.setMinimumSize(160, 240)
        # self.axial_label_image.setMinimumSize(240, 240)
        self.sagittal_image_label.setAlignment(Qt.AlignCenter)
        self.coronal_label_image.setAlignment(Qt.AlignCenter)
        self.axial_label_image.setAlignment(Qt.AlignCenter)

        self.predict_overlap_widget.setFixedWidth(170)

        self.detail_popup = None

        self.show()

    def popup_detail(self):
        eq_sag, eq_cor, eq_axi, ai_sag, ai_cor, ai_axi = self.controller.detail_plots()
        self.detail_popup = SeparateComparison(eq_sag, eq_cor, eq_axi, ai_sag, ai_cor, ai_axi)


    def start_spinner(self):
        self.spinner.start()
    def stop_spinner(self):
        self.spinner.stop()

    def createSpinner(self):
        # Specific info is generated by using pyqtspinner-conf in terminal
        return WaitingSpinner(self.centralwidget, center_on_parent=True, disable_parent_when_spinning=True,
                              roundness=100.0, fade=81.0, radius=15, lines=35, line_length=55, line_width=5,
                              speed=1.5707963267948966, color=QColor(142, 159, 255))

    @override
    def wheelEvent(self, a0):
        moved_angle = a0.angleDelta().y()
        if moved_angle < 0:
            self.resize_image(0.9)
            print("catch mouse scroll down")
        elif moved_angle > 0:
            self.resize_image(1.1)
            print("catch mouse scroll up")

    def resize_image(self, scale_rate):

        sag_width = self.sagittal_image_label.width()
        cor_width = self.coronal_label_image.width()
        axi_width = self.axial_label_image.width()
        sag_height = self.sagittal_image_label.height()
        cor_height = self.coronal_label_image.height()
        axi_height = self.axial_label_image.height()
        print("eq width =", self.equation_widget.width() - self.predict_overlap_widget.width(), "result: ", (sag_width + cor_width + axi_width) * scale_rate)

        if ((sag_width + cor_width + axi_width) * scale_rate < (self.equation_widget.width() - self.predict_overlap_widget.width()) and
            axi_height < self.equation_widget.height() and scale_rate > 1) or (scale_rate < 1 and sag_width > 100):

            self.sagittal_image_label.setFixedHeight(int(sag_height * scale_rate))
            self.sagittal_image_label.setFixedWidth(int(sag_width * scale_rate))

            self.coronal_label_image.setFixedHeight(int(cor_height * scale_rate))
            self.coronal_label_image.setFixedWidth(int(cor_width * scale_rate))

            self.axial_label_image.setFixedHeight(int(axi_height * scale_rate))
            self.axial_label_image.setFixedWidth(int(axi_width * scale_rate))

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
        elif label_key == EquationConstant.T2_KEY:
            self.t2_file_label.setText(file_name)
        elif label_key == EquationConstant.SEG2_KEY:
            self.seg2_file_label.setText(file_name)
        else:
            self.flair2_file_label.setText(file_name)

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
            csf_diff = self.get_diffusion()
            reaction = self.get_reaction()
            grey_diff = self.get_grey_diffusion()
            white_diff = self.get_white_diffusion()
            self.disable_by_start(True)
            self.equation_running_info_label.setText(f"Running Equation Model with CSF diffusion rate {csf_diff},"
                                                     f" white matter diffusion rate {white_diff},\n"
                                                     f"grey matter diffusion rate {grey_diff} and reaction rate {reaction}")
            self.spinner.start()
            self.controller.set_temporal(reaction, csf_diff, grey_diff, white_diff, self.get_cur_scan(),
                                             self.equation_checkBox.isChecked(), self.real_checkBox.isChecked(),
                                             self.ai_checkBox.isChecked(), self.mix_checkBox.isChecked(), self.toggle_checkbox.isChecked())
            self.controller.start()
            # self.controller.start_prediction(reaction, csf_diff, grey_diff, white_diff, self.get_cur_scan(),
            #                                  self.equation_checkBox.isChecked(), self.real_checkBox.isChecked(),
            #                                  self.ai_checkBox.isChecked(), self.mix_checkBox.isChecked(), self.toggle_checkbox.isChecked())

    def update_image_display(self):
        self.controller.update_image_display(self.equation_checkBox.isChecked(), self.real_checkBox.isChecked(),
                                             self.ai_checkBox.isChecked(), self.mix_checkBox.isChecked(), self.toggle_checkbox.isChecked())

    def check_files(self):
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
        elif self.seg2_file_label.text() == "":
            msg = "Missing segment 2 File"
        elif self.flair2_file_label.text() == "":
            msg = "Missing flair 2 File"
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
        # self.set_input_range_label()
        self.set_default_input()

    def save_mask(self):
        self.controller.save_mask(self.slice_slider.value(), self.time_slider.value())

    def set_default_input(self):
        self.diffusion_rate_input.setText(str(EquationConstant.CSF_DIFFUSION_RATE))
        self.grey_diffusion_rate_input.setText(str(EquationConstant.GREY_DIFFUSION_RATE))
        self.white_diffusion_input.setText(str(EquationConstant.WHITE_DIFFUSION_RATE))
        self.reaction_rate_input.setText(str(EquationConstant.REACTION_RATE))

    # def set_input_range_label(self):
    #     self.equation_running_info_label.setText(f"Diffusion Rate Range: [{EquationConstant.MIN_DIFFUSION},{EquationConstant.MAX_DIFFUSION}], "
    #                                              f"Reaction Rate Range: [{EquationConstant.MIN_REACTION}，{EquationConstant.MAX_REACTION}]")

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
        # self.process_info_label.show()
        QApplication.processEvents()
        scan = self.get_cur_scan()
        slice_i = self.slice_slider.value()
        time_i = self.time_slider.value()
        is_overlay = self.toggle_checkbox.isChecked()
        self.controller.process_plots(scan, slice_i, time_i, is_overlay, self.equation_checkBox.isChecked(), self.real_checkBox.isChecked(),
                                      self.ai_checkBox.isChecked(), self.mix_checkBox.isChecked())

    def toggle_overlay(self):
        self.controller.update_image_display(self.equation_checkBox.isChecked(), self.real_checkBox.isChecked(),
                                             self.ai_checkBox.isChecked(), self.mix_checkBox.isChecked())

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
        self.separate_button.setDisabled(not has_start)
        self.disable_input_lineedit(has_start)
        self.disable_file_selection(has_start)
        self.disable_radio_buttons(not has_start)
        self.toggle_checkbox.setDisabled(not has_start)
        self.disable_sliders(not has_start)

    def disable_input_lineedit(self, disable):
        self.diffusion_rate_input.setDisabled(disable)
        self.grey_diffusion_rate_input.setDisabled(disable)
        self.white_diffusion_input.setDisabled(disable)
        self.reaction_rate_input.setDisabled(disable)

    def disable_file_selection(self, disable):
        self.flair_file_button.setDisabled(disable)
        self.glistrboost_file_button.setDisabled(disable)
        self.t1_file_button.setDisabled(disable)
        self.t1gd_file_button.setDisabled(disable)
        self.t2_file_button.setDisabled(disable)
        self.seg_file_button.setDisabled(disable)
        self.flair2_file_button.setDisabled(disable)

    def disable_radio_buttons(self, disable):
        self.flair_rb.setDisabled(disable)
        self.t1_rb.setDisabled(disable)
        self.t1gd_rb.setDisabled(disable)
        self.t2_rb.setDisabled(disable)

    def disable_sliders(self, disable):
        self.time_slider.setDisabled(disable)
        self.slice_slider.setDisabled(disable)

    def update_plot(self, sag, cor, axi):
        sag_height, sag_width, sag_channel = sag.shape
        self.sag_Image = QImage(sag.data, sag_width, sag_height, sag_channel * sag_width, QImage.Format_RGB888)
        self.sagittal_image_label.setPixmap(QPixmap.fromImage(self.sag_Image))

        cor_height, cor_width, cor_channel = cor.shape
        self.cor_Image = QImage(cor.data, cor_width, cor_height, cor_channel * cor_width, QImage.Format_RGB888)
        self.coronal_label_image.setPixmap(QPixmap.fromImage(self.cor_Image))
        axi_height, axi_width, axi_channel = axi.shape
        self.axi_Image = QImage(axi.data, axi_width, axi_height, axi_channel * axi_width, QImage.Format_RGB888)
        self.axial_label_image.setPixmap(QPixmap.fromImage(self.axi_Image))

    def auto_selection(self):
        """
        Selects specific MRI files automatically. Use for testing
        """
        try:
            current_file_path = os.path.dirname(__file__)
            testing_files_path = os.path.join(current_file_path, "100026")
            # testing_files_path = os.path.join(current_file_path, "100001")
            for filename in os.listdir(testing_files_path):
                file_path = os.path.join(testing_files_path, filename)
                if filename.__contains__("time1_flair.nii"):
                    self.update_selected_file_info(EquationConstant.FLAIR_KEY, file_path, filename)
                elif filename.__contains__("time1_seg.nii"):
                    self.update_selected_file_info(EquationConstant.GLISTRBOOST_KEY, file_path, filename)
                elif filename.__contains__("time1_t1.nii"):
                    self.update_selected_file_info(EquationConstant.T1_KEY, file_path, filename)
                elif filename.__contains__("time1_t1ce.nii"):
                    self.update_selected_file_info(EquationConstant.T1GD_KEY, file_path, filename)
                elif filename.__contains__("time1_t2.nii"):
                    self.update_selected_file_info(EquationConstant.T2_KEY, file_path, filename)
                elif filename.__contains__("time2_seg.nii"):
                    self.update_selected_file_info(EquationConstant.SEG2_KEY, file_path, filename)
                elif filename.__contains__("time2_flair.nii"):
                    self.update_selected_file_info("flair 2", file_path, filename)
        except:
            print("Auto Selection Fail!")

    def get_diffusion(self):
        diffusion_rate =EquationConstant.CSF_DIFFUSION_RATE
        try:
            diffusion_rate = float(self.diffusion_rate_input.text())
        except:
            pass
        return diffusion_rate

    def get_reaction(self):
        reaction_rate = EquationConstant.REACTION_RATE
        try:
            reaction_rate = float(self.reaction_rate_input.text())
        except:
            pass
        return reaction_rate

    def get_grey_diffusion(self):
        grey_diffusion_rate = EquationConstant.GREY_DIFFUSION_RATE
        try:
            grey_diffusion_rate = float(self.grey_diffusion_rate_input.text())
        except:
            pass
        return grey_diffusion_rate

    def get_white_diffusion(self):
        white_diffusion_rate = EquationConstant.WHITE_DIFFUSION_RATE
        try:
            white_diffusion_rate = float(self.white_diffusion_rate_input.text())
        except:
            pass
        return white_diffusion_rate

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