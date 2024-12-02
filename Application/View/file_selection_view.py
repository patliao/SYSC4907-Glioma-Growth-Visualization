from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog

from Application.UI_Code.file_selection_ui import Ui_input_widget
from Application.equation_constant import EquationConstant


class FileSelectionView(QtWidgets.QWidget, Ui_input_widget):
    def __init__(self, controller):
        super().__init__()
        self.setupUi(self)
        self.controller = controller
        # print(type(self.flair_file_button))
        self.flair_file_button.clicked.connect(lambda: self.selected_file_clicked(EquationConstant.FLAIR_KEY))
        self.glistrboost_file_button.clicked.connect(lambda: self.selected_file_clicked(EquationConstant.GLISTRBOOST_KEY))
        self.t1_file_button.clicked.connect(lambda: self.selected_file_clicked(EquationConstant.T1_KEY))
        self.t1gd_file_button.clicked.connect(lambda: self.selected_file_clicked(EquationConstant.T1GD_KEY))
        self.t2_file_button.clicked.connect(lambda: self.selected_file_clicked(EquationConstant.T2_KEY))
        self.show()

    def selected_file_clicked(self, button_key):
        file_name, file_path = self.file_select_dialog()
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
        # dlg.setFileMode(QFileDialog.AnyFile)
        # dlg.setFilter("MRI Images (*.nii)")
        # if dlg.exec_():
            # fileName = dlg.selectedFiles()
            # filePath, _ = dlg.getOpenFileName(self, "Open File", "", "All Files (*);;Text Files (*.nii)")
            # pass
        filePath, _ = dlg.getOpenFileName(self, "Open File", "", "All Files (*);;Text Files (*.nii)")
        fileName = filePath.split("/")[-1]
        return fileName, filePath
