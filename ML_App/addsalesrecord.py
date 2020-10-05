from PyQt5.QtWidgets import QWidget,QMessageBox,QApplication
from recordadd_sales import Ui_Form_sales
import pandas as pd
import sqlite3

class AddNewSales(QWidget):
    
    def __init__(self):
        super().__init__()
        
        self.ui = Ui_Form_sales()
        self.ui.setupUi(self)
        
        self.ui.pushButton_add_sales.clicked.connect(self.addsales)
        self.ui.pushButton_clear_sales.clicked.connect(self.clearaddsales)
        
        
    def addsales(self):
        store=str(self.ui.lineEdit_store.text())
        product=str(self.ui.lineEdit_product.text())
        date=str(self.ui.lineEdit_date.text())
        basepric = str(self.ui.lineEdit_baseprice.text())
        price=str(self.ui.lineEdit_price.text())
        weekly=str(self.ui.lineEdit_weekly.text())
        if self.ui.radioButton_TRUE.isChecked():
                        holi = 'TRUE'
        if self.ui.radioButton_FALSE.isChecked():
                        holi = 'FALSE'
        if store == "" or product == "" or date == "" or basepric == "" or price == "" or weekly == "" or holi == "":
            QMessageBox.information(self, "Please Check All Fields!", 'All Fields are reqiured!')
        else:
            try:
                conn = sqlite3.connect('veri_tabanı.db')
                cursor = conn.cursor()
                cursor.execute(
                    'INSERT INTO Stores_Sales(Store,Product,Date,Is_Holiday,BasePrice,Price,Weekly_Units_Sold) VALUES (?,?,?,?,?,?,?)',
                    (store,product,date,holi,basepric,price,weekly))
                conn.commit()
                cursor.close()
                conn.close()
                QMessageBox.information(self, "Success!", 'New sales record created Successfully!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Please check the entries..')
    
     
    def clearaddsales(self):
        self.ui.lineEdit_store.setText('')
        self.ui.lineEdit_product.setText('')
        self.ui.lineEdit_date.setText('')
        self.ui.lineEdit_price.setText('')
        self.ui.lineEdit_weekly.setText('')
