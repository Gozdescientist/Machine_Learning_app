from PyQt5.QtWidgets import QWidget,QMessageBox,QApplication
from recordadd_retail import Ui_Form_retail
import pandas as pd
import sqlite3

class AddNewRetail(QWidget):
    
    def __init__(self):
        super().__init__()
        
        self.ui = Ui_Form_retail()
        self.ui.setupUi(self)
        
        
        self.ui.pushButton_add.clicked.connect(self.addretail)
        self.ui.pushButton_clear.clicked.connect(self.clearaddretail)
        conn = sqlite3.connect('veri_tabanı.db')
        df = pd.read_sql_query("select distinct Country from OnlineRetails",conn)
        combolist = df['Country'].tolist()
        self.ui.comboBox_country.addItems(combolist)
        
    def addretail(self):
        invo = str(self.ui.lineEdit_invoiceno.text())
        stockc = str(self.ui.lineEdit_stockcode.text())
        descr = str(self.ui.lineEdit_stockdesc.text())
        quant = str(self.ui.lineEdit_quantity.text())
        invodate = str(self.ui.lineEdit_invoicedate.text())
        uprice = str(self.ui.lineEdit_unitprice.text())
        customer = str(self.ui.lineEdit_customerid.text())
        country = str(self.ui.comboBox_country.currentText())
        if invo == "" or stockc == "" or descr == "" or quant == "" or invodate == "" or uprice == "" or customer == "" or country == "":
            QMessageBox.information(self, "Please Check All Fields!", 'All Fields are reqiured!')
        else:
            try:
                conn = sqlite3.connect('veri_tabanı.db')
                cursor = conn.cursor()
                cursor.execute(
                        'INSERT INTO OnlineRetails(InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country) VALUES (?,?,?,?,?,?,?,?)',
                        (invo,stockc,descr,quant,invodate,uprice,customer,country))
                conn.commit()
                cursor.close()
                conn.close()
                QMessageBox.information(self, "Success!", 'New online retail record created Successfully!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Please check the entries..')
                
    def clearaddretail(self):
         self.ui.lineEdit_invoiceno.setText('')
         self.ui.lineEdit_stockcode.setText('')
         self.ui.lineEdit_stockdesc.setText('')
         self.ui.lineEdit_quantity.setText('')
         self.ui.lineEdit_invoicedate.setText('')
         self.ui.lineEdit_unitprice.setText('')
         self.ui.lineEdit_customerid.setText('')
         

