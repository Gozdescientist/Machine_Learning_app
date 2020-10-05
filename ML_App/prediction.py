# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'prediction.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1066, 694)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/icons/bars.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("#centralwidget{\n"
"border-image: url(:/icons/icons/main.png);\n"
"}")
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Segoe Print")
        font.setPointSize(25)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_4.addWidget(self.label_3)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        spacerItem1 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.formFrame = QtWidgets.QFrame(self.centralwidget)
        self.formFrame.setMinimumSize(QtCore.QSize(400, 220))
        self.formFrame.setMaximumSize(QtCore.QSize(16777215, 100))
        self.formFrame.setStyleSheet("#formFrame{\n"
"background-color: rgb(255, 255, 255);\n"
"border-radius: 10px\n"
"}")
        self.formFrame.setObjectName("formFrame")
        self.formLayout = QtWidgets.QFormLayout(self.formFrame)
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(self.formFrame)
        self.label.setMaximumSize(QtCore.QSize(16777215, 300))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label.setObjectName("label")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.label)
        self.lineEdit_username = QtWidgets.QLineEdit(self.formFrame)
        self.lineEdit_username.setMinimumSize(QtCore.QSize(0, 30))
        self.lineEdit_username.setObjectName("lineEdit_username")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.lineEdit_username)
        self.label_2 = QtWidgets.QLabel(self.formFrame)
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.label_2)
        self.lineEdit_password = QtWidgets.QLineEdit(self.formFrame)
        self.lineEdit_password.setMinimumSize(QtCore.QSize(0, 30))
        self.lineEdit_password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.lineEdit_password.setObjectName("lineEdit_password")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.lineEdit_password)
        self.label_4 = QtWidgets.QLabel(self.formFrame)
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.FieldRole, self.label_4)
        self.PushButton_signup = QtWidgets.QPushButton(self.formFrame)
        self.PushButton_signup.setMinimumSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.PushButton_signup.setFont(font)
        self.PushButton_signup.setStyleSheet("color: rgb(0, 0, 0);\n"
"border-right-color: rgb(0, 0, 0);\n"
"border-color: rgb(85, 0, 255);\n"
"background-color: rgb(174, 229, 183);\n"
"border-radius: 10px\n"
"")
        self.PushButton_signup.setObjectName("PushButton_signup")
        self.formLayout.setWidget(8, QtWidgets.QFormLayout.FieldRole, self.PushButton_signup)
        self.horizontalLayout_2.addWidget(self.formFrame)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem2)
        self.verticalLayout_4.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)
        spacerItem4 = QtWidgets.QSpacerItem(20, 120, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.horizontalLayout.addItem(spacerItem4)
        self.PushButton_login = QtWidgets.QPushButton(self.centralwidget)
        self.PushButton_login.setMinimumSize(QtCore.QSize(150, 70))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.PushButton_login.setFont(font)
        self.PushButton_login.setStyleSheet("color: rgb(0, 0, 0);\n"
"border-right-color: rgb(0, 0, 0);\n"
"border-color: rgb(85, 0, 255);\n"
"background-color: rgb(75, 150, 225);\n"
"border-radius: 10px\n"
"")
        self.PushButton_login.setObjectName("PushButton_login")
        self.horizontalLayout.addWidget(self.PushButton_login)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem5)
        self.verticalLayout_4.addLayout(self.horizontalLayout)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(8)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setStyleSheet("")
        self.label_5.setObjectName("label_5")
        self.verticalLayout_4.addWidget(self.label_5)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1066, 26))
        self.menubar.setObjectName("menubar")
        self.menuApplication = QtWidgets.QMenu(self.menubar)
        self.menuApplication.setObjectName("menuApplication")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.menuApplication.addAction(self.actionExit)
        self.menubar.addAction(self.menuApplication.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Welcome!!"))
        self.label_3.setText(_translate("MainWindow", "Machine Learning Predictions"))
        self.label.setText(_translate("MainWindow", "Username"))
        self.label_2.setText(_translate("MainWindow", "Password"))
        self.label_4.setText(_translate("MainWindow", "Don\'t have an account?"))
        self.PushButton_signup.setText(_translate("MainWindow", "SignUp"))
        self.PushButton_login.setText(_translate("MainWindow", "Login"))
        self.label_5.setText(_translate("MainWindow", "This application aims to analyze the business processes of different departments of the Group Company and obtain various predictions.            All rights reserved."))
        self.menuApplication.setTitle(_translate("MainWindow", "Application"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
import icons_rc


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
