
from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap

from detailDialogView import Ui_prediction_detail
class SeparateComparison(QtWidgets.QDialog, Ui_prediction_detail):
    def __init__(self, eq_sag, eq_cor, eq_axi, ai_sag, ai_cor, ai_axi):
        super(SeparateComparison, self).__init__()
        self.setupUi(self)

        # equation
        eq_sag_height, eq_sag_width, eq_sag_channel = eq_sag.shape
        eq_sag_Image = QImage(eq_sag.data, eq_sag_width, eq_sag_height, eq_sag_channel * eq_sag_width, QImage.Format_RGB888)
        self.eq_sag_label.setPixmap(QPixmap.fromImage(eq_sag_Image))
        eq_cor_height, eq_cor_width, eq_cor_channel = eq_cor.shape
        eq_cor_Image = QImage(eq_cor.data, eq_cor_width, eq_cor_height, eq_cor_channel * eq_cor_width, QImage.Format_RGB888)
        self.eq_cor_label.setPixmap(QPixmap.fromImage(eq_cor_Image))
        eq_axi_height, eq_axi_width, eq_axi_channel = eq_axi.shape
        eq_axi_Image = QImage(eq_axi.data, eq_axi_width, eq_axi_height, eq_axi_channel * eq_axi_width, QImage.Format_RGB888)
        self.eq_axi_label.setPixmap(QPixmap.fromImage(eq_axi_Image))
        # ai
        ai_sag_height, ai_sag_width, ai_sag_channel = ai_sag.shape
        ai_sag_Image = QImage(ai_sag.data, ai_sag_width, ai_sag_height, ai_sag_channel * ai_sag_width, QImage.Format_RGB888)
        self.ai_sag_label.setPixmap(QPixmap.fromImage(ai_sag_Image))
        ai_cor_height, ai_cor_width, ai_cor_channel = ai_cor.shape
        ai_cor_Image = QImage(ai_cor.data, ai_cor_width, ai_cor_height, ai_cor_channel * ai_cor_width, QImage.Format_RGB888)
        self.ai_cor_label.setPixmap(QPixmap.fromImage(ai_cor_Image))
        ai_axi_height, ai_axi_width, ai_axi_channel = ai_axi.shape
        ai_axi_Image = QImage(ai_axi.data, ai_axi_width, ai_axi_height, ai_axi_channel * ai_axi_width, QImage.Format_RGB888)
        self.ai_axi_label.setPixmap(QPixmap.fromImage(ai_axi_Image))





        self.show()