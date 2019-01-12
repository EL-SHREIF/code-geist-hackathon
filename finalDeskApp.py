# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'finalDeskApp.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui
import sys
import subprocess

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_DeskApp(object):
    def setupUi(self, DeskApp):
        DeskApp.setObjectName(_fromUtf8("DeskApp"))
        DeskApp.resize(561, 555)
        self.centralwidget = QtGui.QWidget(DeskApp)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.frame = QtGui.QFrame(self.centralwidget)
        self.frame.setStyleSheet(_fromUtf8("background-color: rgb(0, 0, 0);"))
        self.frame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtGui.QFrame.Raised)
        self.frame.setObjectName(_fromUtf8("frame"))
        self.verticalLayoutWidget = QtGui.QWidget(self.frame)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(-50, 339, 601, 191))
        self.verticalLayoutWidget.setObjectName(_fromUtf8("verticalLayoutWidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.frame_2 = QtGui.QFrame(self.verticalLayoutWidget)
        self.frame_2.setStyleSheet(_fromUtf8("background-color: rgb(212, 175, 55);"))
        self.frame_2.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_2.setObjectName(_fromUtf8("frame_2"))
        self.soon = QtGui.QPushButton(self.frame_2)
        self.soon.setGeometry(QtCore.QRect(220, 50, 211, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.soon.setFont(font)
        self.soon.setStyleSheet(_fromUtf8("background-color: rgb(0, 0, 0);\n"
"color: rgb(212, 175, 55);"))
        self.soon.setObjectName(_fromUtf8("soon"))
        self.verticalLayout.addWidget(self.frame_2)
        self.Sim = QtGui.QPushButton(self.frame)
        self.Sim.setGeometry(QtCore.QRect(170, 30, 211, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.Sim.setFont(font)
        self.Sim.setStyleSheet(_fromUtf8("background-color: rgb(212, 175, 55);"))
        self.Sim.setObjectName(_fromUtf8("Sim"))
        self.SL = QtGui.QPushButton(self.frame)
        self.SL.setGeometry(QtCore.QRect(170, 200, 211, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.SL.setFont(font)
        self.SL.setStyleSheet(_fromUtf8("background-color: rgb(212, 175, 55);"))
        self.SL.setObjectName(_fromUtf8("SL"))
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)
        DeskApp.setCentralWidget(self.centralwidget)
        self.statusbar = QtGui.QStatusBar(DeskApp)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        DeskApp.setStatusBar(self.statusbar)
        self.menuBar = QtGui.QMenuBar(DeskApp)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 561, 22))
        self.menuBar.setObjectName(_fromUtf8("menuBar"))
        DeskApp.setMenuBar(self.menuBar)

        self.retranslateUi(DeskApp)
        QtCore.QObject.connect(self.soon, QtCore.SIGNAL(_fromUtf8("clicked()")), self.SL.click)
        QtCore.QObject.connect(self.SL, QtCore.SIGNAL(_fromUtf8("clicked()")), openshiko)
        QtCore.QObject.connect(self.Sim, QtCore.SIGNAL(_fromUtf8("clicked()")), self.SL.click)
        QtCore.QMetaObject.connectSlotsByName(DeskApp)

    def retranslateUi(self, DeskApp):
        DeskApp.setWindowTitle(_translate("DeskApp", "MainWindow", None))
        self.soon.setText(_translate("DeskApp", "SOON...", None))
        self.Sim.setText(_translate("DeskApp", "SimGame", None))
        self.SL.setText(_translate("DeskApp", "SLTaxi", None))

def openshiko():
    subprocess.Popen("main.py 1", shell=True)


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    DeskApp = QtGui.QMainWindow()
    ui = Ui_DeskApp()
    ui.setupUi(DeskApp)
    DeskApp.show()
    sys.exit(app.exec_())

