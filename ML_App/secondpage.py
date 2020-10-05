from PyQt5.QtWidgets import *
from register import Ui_Dialog
import re
import sqlite3
import datetime

class SecondPage(QDialog):
    
    def __init__(self):
        super().__init__()
        
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        
        self.ui.pushButton_create.clicked.connect(self.register)
        self.ui.pushButton_clear.clicked.connect(self.clear1)
        
        
    def clear1(self):
        self.ui.lineEdit_register.setText('')
        self.ui.lineEdit_name.setText('')
        self.ui.lineEdit_surname.setText('')
        self.ui.lineEdit_age.setText('')
        self.ui.lineEdit_username.setText('')
        self.ui.lineEdit_password.setText('')
        self.ui.lineEdit_email.setText('')
        self.ui.lineEdit_mobile.setText('')
        self.ui.comboBox_dept.setCurrentText('')
        self.ui.radioButton_female.setAutoExclusive(False)
        self.ui.radioButton_female.setChecked(False)
        self.ui.radioButton_female.setAutoExclusive(True)
        self.ui.radioButton_male.setAutoExclusive(False)
        self.ui.radioButton_male.setChecked(False)
        self.ui.radioButton_male.setAutoExclusive(True)
        


    def register(self):
            regexmail = '^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$'
            regexage = '(1[89]|[2-6][0-9]|70)'
            regexpassword = '^(?=.*[0-9])(?=.*[a-z])(?=.*[A-Z])(?=.*[@#$%^&+=.\-_*])([a-zA-Z0-9@#$%^&+=*.\-_]){3,}$'
            regexusername = '^[a-zA-Z]{3,16}$'
            regexletter = '^[a-zA-Z]+$'
            regexpositivenum = '^\d+$'
            regexregister = '^(?=.*[A-Z])(?=.*[0-9])'
            password = self.ui.lineEdit_password.text()
            email = self.ui.lineEdit_email.text()
            age = self.ui.lineEdit_age.text()
            username = self.ui.lineEdit_username.text()
            name = self.ui.lineEdit_name.text()
            surname = self.ui.lineEdit_surname.text()
            mobile = self.ui.lineEdit_mobile.text()
            register = self.ui.lineEdit_register.text()
            matchObje = re.search(regexmail,email)
            matchObja = re.search(regexage,age)
            matchObjp = re.search(regexpassword,password)
            matchObju = re.search(regexusername,username)
            matchObjn = re.search(regexletter,name)
            matchObjs = re.search(regexletter,surname)
            matchOnjp = re.search(regexpositivenum,mobile)
            matchOnjr = re.search(regexregister,register)
            if password == "" or email == "" or age == "" or username == "" or name == "" or surname == "" or mobile == "" or register == "":
                QMessageBox.information(self, "Please Check All Fields!", 'All Fields are reqiured!')
            elif not matchObje:
                QMessageBox.information(self, "Invalid e-Mail!", 'Please check email entries again..')
            elif not matchObja:
                QMessageBox.information(self, "Invalid Age!", 'Age should be between 18 and 70')
            elif not matchObjp:
                QMessageBox.information(self, "Invalid Password!", 'Password must contain a number, a lowercase, a uppercase and a special character: such as Ab12$ and Bf7#')
            elif not matchObju:
                QMessageBox.information(self, "Invalid Username!", 'Username must contain only a lowercase and uppercase not a number')
            elif not matchObjn:
                QMessageBox.information(self, "Invalid Name!", 'Name only contain letters')
            elif not matchObjs:
                QMessageBox.information(self, "Invalid Surname!", 'Surname only contain letters')
            elif not matchOnjp:
                QMessageBox.information(self, "Invalid Mobile Phone!", 'Mobile Phone must contain only numbers')
            elif not matchOnjr:
                QMessageBox.information(self, "Invalid Register Number!", 'Register Number must contain uppercase and number')
            else:
                try:
                    register=str(self.ui.lineEdit_register.text())
                    name = str(self.ui.lineEdit_name.text())
                    surname = str(self.ui.lineEdit_surname.text())
                    age = int(self.ui.lineEdit_age.text())
                    username= str(self.ui.lineEdit_username.text())
                    password = str(self.ui.lineEdit_password.text())
                    department = str(self.ui.comboBox_dept.currentText())
                    email= str(self.ui.lineEdit_email.text())
                    mobile = int(self.ui.lineEdit_mobile.text())
                    dtime = str(datetime.datetime.now())
                    gender = None
            
                    if self.ui.radioButton_male.isChecked():
                        gender = 'Male'
                    if self.ui.radioButton_female.isChecked():
                        gender = 'Female'

                    conn = sqlite3.connect('veri_tabanı.db')
                    cursor = conn.cursor()
                    cursor.execute(
                            'INSERT INTO UserAccounts(RegisterNumber,Name,Surname,Age,Username,Password,Department,Email,Mobile,RecordDate,Gender) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                            (register, name, surname,age ,username, password, department,email ,mobile, dtime, gender))
                    conn.commit()
                    cursor.close()
                    conn.close()
                    QMessageBox.information(self, "Success!", 'New User Account created Successfully!')
                except sqlite3.Error as error:
                    QMessageBox.critical(self, "Error!", 'Please check the Register Number! This number belongs to another user..')