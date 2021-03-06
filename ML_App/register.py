# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'register.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(706, 782)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/icons/user.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Dialog.setWindowIcon(icon)
        Dialog.setStyleSheet("background-color: rgb(211, 236, 218);")
        self.pushButton_clear = QtWidgets.QPushButton(Dialog)
        self.pushButton_clear.setGeometry(QtCore.QRect(370, 670, 271, 41))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_clear.setFont(font)
        self.pushButton_clear.setStyleSheet("background-color: rgb(77, 155, 232);\n"
"border-radius: 10px")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/icons/clear.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_clear.setIcon(icon1)
        self.pushButton_clear.setIconSize(QtCore.QSize(50, 40))
        self.pushButton_clear.setObjectName("pushButton_clear")
        self.label_9 = QtWidgets.QLabel(Dialog)
        self.label_9.setGeometry(QtCore.QRect(240, 30, 231, 61))
        font = QtGui.QFont()
        font.setFamily("Segoe Print")
        font.setPointSize(22)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.pushButton_create = QtWidgets.QPushButton(Dialog)
        self.pushButton_create.setGeometry(QtCore.QRect(60, 670, 291, 41))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_create.setFont(font)
        self.pushButton_create.setStyleSheet("background-color: rgb(248, 184, 120);\n"
"border-radius: 10px")
        self.pushButton_create.setIcon(icon)
        self.pushButton_create.setIconSize(QtCore.QSize(53, 40))
        self.pushButton_create.setObjectName("pushButton_create")
        self.formFrame = QtWidgets.QFrame(Dialog)
        self.formFrame.setGeometry(QtCore.QRect(110, 110, 501, 541))
        self.formFrame.setStyleSheet("")
        self.formFrame.setObjectName("formFrame")
        self.formLayout = QtWidgets.QFormLayout(self.formFrame)
        self.formLayout.setContentsMargins(7, 7, 7, 7)
        self.formLayout.setHorizontalSpacing(20)
        self.formLayout.setVerticalSpacing(15)
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(self.formFrame)
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.lineEdit_register = QtWidgets.QLineEdit(self.formFrame)
        self.lineEdit_register.setMinimumSize(QtCore.QSize(0, 30))
        self.lineEdit_register.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.lineEdit_register.setObjectName("lineEdit_register")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_register)
        self.label_4 = QtWidgets.QLabel(self.formFrame)
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.lineEdit_name = QtWidgets.QLineEdit(self.formFrame)
        self.lineEdit_name.setMinimumSize(QtCore.QSize(0, 30))
        self.lineEdit_name.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.lineEdit_name.setObjectName("lineEdit_name")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_name)
        self.label_5 = QtWidgets.QLabel(self.formFrame)
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.lineEdit_surname = QtWidgets.QLineEdit(self.formFrame)
        self.lineEdit_surname.setMinimumSize(QtCore.QSize(0, 30))
        self.lineEdit_surname.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.lineEdit_surname.setObjectName("lineEdit_surname")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.lineEdit_surname)
        self.label_12 = QtWidgets.QLabel(self.formFrame)
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_12)
        self.lineEdit_age = QtWidgets.QLineEdit(self.formFrame)
        self.lineEdit_age.setMinimumSize(QtCore.QSize(0, 30))
        self.lineEdit_age.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.lineEdit_age.setObjectName("lineEdit_age")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.lineEdit_age)
        self.label_6 = QtWidgets.QLabel(self.formFrame)
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.comboBox_dept = QtWidgets.QComboBox(self.formFrame)
        self.comboBox_dept.setMinimumSize(QtCore.QSize(0, 30))
        self.comboBox_dept.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.comboBox_dept.setObjectName("comboBox_dept")
        self.comboBox_dept.addItem("")
        self.comboBox_dept.addItem("")
        self.comboBox_dept.addItem("")
        self.comboBox_dept.addItem("")
        self.comboBox_dept.addItem("")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.comboBox_dept)
        self.label_2 = QtWidgets.QLabel(self.formFrame)
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.lineEdit_username = QtWidgets.QLineEdit(self.formFrame)
        self.lineEdit_username.setMinimumSize(QtCore.QSize(0, 30))
        self.lineEdit_username.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.lineEdit_username.setObjectName("lineEdit_username")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.FieldRole, self.lineEdit_username)
        self.label_3 = QtWidgets.QLabel(self.formFrame)
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(8, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.lineEdit_password = QtWidgets.QLineEdit(self.formFrame)
        self.lineEdit_password.setMinimumSize(QtCore.QSize(0, 30))
        self.lineEdit_password.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.lineEdit_password.setObjectName("lineEdit_password")
        self.formLayout.setWidget(8, QtWidgets.QFormLayout.FieldRole, self.lineEdit_password)
        self.label_7 = QtWidgets.QLabel(self.formFrame)
        self.label_7.setText("")
        self.label_7.setObjectName("label_7")
        self.formLayout.setWidget(9, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.label_10 = QtWidgets.QLabel(self.formFrame)
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.formLayout.setWidget(10, QtWidgets.QFormLayout.LabelRole, self.label_10)
        self.lineEdit_email = QtWidgets.QLineEdit(self.formFrame)
        self.lineEdit_email.setMinimumSize(QtCore.QSize(0, 30))
        self.lineEdit_email.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.lineEdit_email.setObjectName("lineEdit_email")
        self.formLayout.setWidget(10, QtWidgets.QFormLayout.FieldRole, self.lineEdit_email)
        self.lineEdit_mobile = QtWidgets.QLineEdit(self.formFrame)
        self.lineEdit_mobile.setMinimumSize(QtCore.QSize(0, 30))
        self.lineEdit_mobile.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.lineEdit_mobile.setObjectName("lineEdit_mobile")
        self.formLayout.setWidget(11, QtWidgets.QFormLayout.FieldRole, self.lineEdit_mobile)
        self.label_11 = QtWidgets.QLabel(self.formFrame)
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.formLayout.setWidget(14, QtWidgets.QFormLayout.LabelRole, self.label_11)
        self.label_8 = QtWidgets.QLabel(self.formFrame)
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.formLayout.setWidget(11, QtWidgets.QFormLayout.LabelRole, self.label_8)
        self.radioButton_female = QtWidgets.QRadioButton(self.formFrame)
        self.radioButton_female.setObjectName("radioButton_female")
        self.formLayout.setWidget(14, QtWidgets.QFormLayout.FieldRole, self.radioButton_female)
        self.radioButton_male = QtWidgets.QRadioButton(self.formFrame)
        self.radioButton_male.setObjectName("radioButton_male")
        self.formLayout.setWidget(15, QtWidgets.QFormLayout.FieldRole, self.radioButton_male)
        self.label_13 = QtWidgets.QLabel(Dialog)
        self.label_13.setGeometry(QtCore.QRect(50, 730, 651, 21))
        font = QtGui.QFont()
        font.setFamily("Comic Sans MS")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_13.setFont(font)
        self.label_13.setStyleSheet("color: rgb(0, 0, 0);\n"
"")
        self.label_13.setObjectName("label_13")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Register!!"))
        self.pushButton_clear.setText(_translate("Dialog", "Clear"))
        self.label_9.setText(_translate("Dialog", "Registration"))
        self.pushButton_create.setText(_translate("Dialog", "Register"))
        self.label.setText(_translate("Dialog", "Comp. Register Number"))
        self.lineEdit_register.setToolTip(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">Register Number must contain only uppercase and number!</span></p></body></html>"))
        self.label_4.setText(_translate("Dialog", "Name"))
        self.lineEdit_name.setToolTip(_translate("Dialog", "<html><head/><body><p align=\"justify\"><span style=\" font-weight:600;\">Name only contain letters!</span></p></body></html>"))
        self.label_5.setText(_translate("Dialog", "Surname"))
        self.lineEdit_surname.setToolTip(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">Surname only contain letters!</span></p></body></html>"))
        self.label_12.setText(_translate("Dialog", "Age"))
        self.lineEdit_age.setToolTip(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">Age should be between 18 and 70!</span></p></body></html>"))
        self.label_6.setText(_translate("Dialog", "Department"))
        self.comboBox_dept.setItemText(0, _translate("Dialog", "Information Systems"))
        self.comboBox_dept.setItemText(1, _translate("Dialog", "Production"))
        self.comboBox_dept.setItemText(2, _translate("Dialog", "Marketing"))
        self.comboBox_dept.setItemText(3, _translate("Dialog", "R&D"))
        self.comboBox_dept.setItemText(4, _translate("Dialog", "Sales Operation"))
        self.label_2.setText(_translate("Dialog", "Username"))
        self.lineEdit_username.setToolTip(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">Username can be only lowercase and uppercase, to be at least two characters long!</span></p><p><span style=\" font-weight:600;\">Examples : Uname, Suser ,Bref</span></p><p><br/></p></body></html>"))
        self.label_3.setText(_translate("Dialog", "Password"))
        self.lineEdit_password.setToolTip(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">Password must contain a number, a lowercase, a uppercase and a special character!</span></p><p><span style=\" font-weight:600;\">Examples: Ab12$ , Bf7# , Pea9&amp;</span></p></body></html>"))
        self.label_10.setText(_translate("Dialog", "e-Mail"))
        self.lineEdit_email.setToolTip(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">Mail format examples:</span></p><p><span style=\" font-weight:600;\">loginuser@register.com</span></p><p><span style=\" font-weight:600;\">register@loginuser.com.tr</span></p></body></html>"))
        self.lineEdit_mobile.setToolTip(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">Mobile Phone must contain only numbers!</span></p></body></html>"))
        self.label_11.setText(_translate("Dialog", "Gender"))
        self.label_8.setText(_translate("Dialog", "Mobile Phone"))
        self.radioButton_female.setText(_translate("Dialog", "Female"))
        self.radioButton_male.setText(_translate("Dialog", "Male"))
        self.label_13.setText(_translate("Dialog", "Please check tooltips before register, which is shown when you hover over the entry fields!"))
import icons_rc


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
