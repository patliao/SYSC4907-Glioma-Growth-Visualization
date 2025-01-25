import sys

from PyQt5 import QtWidgets

from main_window_controller import MainWindowController

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindowController.instance()
    sys.exit(app.exec_())
