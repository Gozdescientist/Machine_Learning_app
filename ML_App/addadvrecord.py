from PyQt5.QtWidgets import QWidget,QMessageBox,QApplication
from recordadd import Ui_Form
import pandas as pd
import sqlite3

class AddNewAdver(QWidget):
    
    def __init__(self):
        super().__init__()
        
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        
        self.ui.pushButton_add.clicked.connect(self.addadvertising)
        
    def addadvertising(self):
        num=str(self.ui.lineEdit_num.text())
        tv=str(self.ui.lineEdit_tv.text())
        radio=str(self.ui.lineEdit_rad.text())
        news=str(self.ui.lineEdit_news.text())
        total=str(self.ui.lineEdit_total.text())
        if num == "" or tv == "" or radio == "" or news == "" or total == "":
            QMessageBox.information(self, "Please Check All Fields!", 'All Fields are reqiured!')
        else:
            try:    
                conn = sqlite3.connect('veri_tabanı.db')
                cursor = conn.cursor()
                cursor.execute(
                        'INSERT INTO Advertising(field1,TV,radio,newspaper,sales) VALUES (?,?,?,?,?)',
                        (num,tv,radio,news,total))
                conn.commit()
                cursor.close()
                conn.close()
                QMessageBox.information(self, "Success!", 'New advertising record created Successfully!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Please check the entries..')
        
        
      
    

