# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'detailDialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_prediction_detail(object):
    def setupUi(self, prediction_detail):
        prediction_detail.setObjectName("prediction_detail")
        prediction_detail.resize(400, 300)
        self.verticalLayout = QtWidgets.QVBoxLayout(prediction_detail)
        self.verticalLayout.setObjectName("verticalLayout")
        self.eq_label = QtWidgets.QLabel(prediction_detail)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.eq_label.sizePolicy().hasHeightForWidth())
        self.eq_label.setSizePolicy(sizePolicy)
        self.eq_label.setObjectName("eq_label")
        self.verticalLayout.addWidget(self.eq_label)
        self.eq_images_widget = QtWidgets.QWidget(prediction_detail)
        self.eq_images_widget.setObjectName("eq_images_widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.eq_images_widget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.eq_sag_label = QtWidgets.QLabel(self.eq_images_widget)
        self.eq_sag_label.setText("")
        self.eq_sag_label.setObjectName("eq_sag_label")
        self.horizontalLayout.addWidget(self.eq_sag_label)
        self.eq_cor_label = QtWidgets.QLabel(self.eq_images_widget)
        self.eq_cor_label.setText("")
        self.eq_cor_label.setObjectName("eq_cor_label")
        self.horizontalLayout.addWidget(self.eq_cor_label)
        self.eq_axi_label = QtWidgets.QLabel(self.eq_images_widget)
        self.eq_axi_label.setText("")
        self.eq_axi_label.setObjectName("eq_axi_label")
        self.horizontalLayout.addWidget(self.eq_axi_label)
        self.verticalLayout.addWidget(self.eq_images_widget)
        self.aai_label = QtWidgets.QLabel(prediction_detail)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.aai_label.sizePolicy().hasHeightForWidth())
        self.aai_label.setSizePolicy(sizePolicy)
        self.aai_label.setObjectName("aai_label")
        self.verticalLayout.addWidget(self.aai_label)
        self.ai_images_widget = QtWidgets.QWidget(prediction_detail)
        self.ai_images_widget.setObjectName("ai_images_widget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.ai_images_widget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.ai_sag_label = QtWidgets.QLabel(self.ai_images_widget)
        self.ai_sag_label.setText("")
        self.ai_sag_label.setObjectName("ai_sag_label")
        self.horizontalLayout_2.addWidget(self.ai_sag_label)
        self.ai_cor_label = QtWidgets.QLabel(self.ai_images_widget)
        self.ai_cor_label.setText("")
        self.ai_cor_label.setObjectName("ai_cor_label")
        self.horizontalLayout_2.addWidget(self.ai_cor_label)
        self.ai_axi_label = QtWidgets.QLabel(self.ai_images_widget)
        self.ai_axi_label.setText("")
        self.ai_axi_label.setObjectName("ai_axi_label")
        self.horizontalLayout_2.addWidget(self.ai_axi_label)
        self.verticalLayout.addWidget(self.ai_images_widget)

        self.retranslateUi(prediction_detail)
        QtCore.QMetaObject.connectSlotsByName(prediction_detail)

    def retranslateUi(self, prediction_detail):
        _translate = QtCore.QCoreApplication.translate
        prediction_detail.setWindowTitle(_translate("prediction_detail", "Results"))
        self.eq_label.setText(_translate("prediction_detail", "Equation:"))
        self.aai_label.setText(_translate("prediction_detail", "AI:"))
