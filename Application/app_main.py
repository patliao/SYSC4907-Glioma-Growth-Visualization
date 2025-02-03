import sys

from PyQt6 import QtWidgets

from Application.main_window_controller import MainWindowController

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindowController.instance()
    sys.exit(app.exec())
