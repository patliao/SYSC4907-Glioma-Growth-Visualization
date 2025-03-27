import sys
from PyQt5 import QtWidgets

from main_window_view import MainWindowView

if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    MainWindowView()
    sys.exit(app.exec_())
