from PyQt5.QtWidgets import QMessageBox,QApplication,QMainWindow,QFileDialog
from prediction import Ui_MainWindow
from secondpage import SecondPage
from reportscreen import ReportMain
import pandas as pd
import sqlite3
import datetime
import sys


class MainPage(QMainWindow):
    
    def __init__(self):
        super().__init__()
        
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.ui.PushButton_signup.clicked.connect(self.register)
        self.ui.PushButton_login.clicked.connect(self.login)
        self.ui.actionExit.triggered.connect(self.ex)
        
        self.secondpage = SecondPage()
        self.reportscreen = ReportMain()
    
    def register(self):
        
        self.secondpage.show()
    
    
    def login(self):
        with sqlite3.connect('veri_tabanı.db') as db:
            c = db.cursor()

            usernamelog = str(self.ui.lineEdit_username.text())
            passwordlog = str(self.ui.lineEdit_password.text())
            c.execute('Select Name,Surname from UserAccounts where Username = ? and Password = ?', (usernamelog, passwordlog))
            data = c.fetchone()
            db.commit()
            if data != None:
                ns = "Welcome "+ str(data[0]) +" "+ str(data[1])+" :) Nice to see you!"
                conn = sqlite3.connect('veri_tabanı.db')
                QMessageBox.information(window, "Success!",ns)
                self.reportscreen.show()
                window.hide()
                
                
    def ex(self):
        sys.exit()
                
    

                
app = QApplication([])
window = MainPage()
window.show()
app.exec_()
                
               
        
        
        
        
        
        
        