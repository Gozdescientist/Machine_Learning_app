from PyQt5.QtWidgets import QWidget,QMessageBox,QApplication
from recordadd_wine import Ui_Form_wine
import pandas as pd
import sqlite3

class AddNewWine(QWidget):
    
    def __init__(self):
        super().__init__()
        
        self.ui = Ui_Form_wine()
        self.ui.setupUi(self)
        
        self.ui.pushButton_add_wine.clicked.connect(self.addwine)
        self.ui.pushButton_clear_wine.clicked.connect(self.clearaddwine)
        
    
    def addwine(self):
        fixa = str(self.ui.lineEdit_fix.text())
        volat = str(self.ui.lineEdit_vol.text())
        cit = str(self.ui.lineEdit_cit.text())
        sugar = str(self.ui.lineEdit_res.text())
        chlo = str(self.ui.lineEdit_chl.text())
        free = str(self.ui.lineEdit_free.text())
        total = str(self.ui.lineEdit_totalsul.text())
        denst = str(self.ui.lineEdit_densi.text())
        ph = str(self.ui.lineEdit_ph.text())
        suph = str(self.ui.lineEdit_sulp.text())
        alch = str(self.ui.lineEdit_alco.text())
        qual = str(self.ui.lineEdit_qual.text())
        if fixa == "" or volat == "" or cit == "" or sugar == "" or chlo == "" or free == "" or total == "" or denst == "" or ph == "" or suph == "" or alch == "" or qual == "":
            QMessageBox.information(self, "Please Check All Fields!", 'All Fields are reqiured!')
        else:
            try:
                conn = sqlite3.connect('veri_tabanı.db')
                cursor = conn.cursor()
                cursor.execute(
                        'INSERT INTO Wine(fixedacidity,volatileacidity,citricacid,residualsugar,chlorides,freesulfurdioxide,totalsulfurdioxide,density,pH,sulphates,alcohol,quality) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                        (fixa,volat,cit,sugar,chlo,free,total,denst,ph,suph,alch,qual))
                conn.commit()
                cursor.close()
                conn.close()
                QMessageBox.information(self, "Success!", 'New wine experiment record created Successfully!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Please check the entries..')
                
    def clearaddwine(self):
         self.ui.lineEdit_fix.setText('')
         self.ui.lineEdit_vol.setText('')
         self.ui.lineEdit_cit.setText('')
         self.ui.lineEdit_res.setText('')
         self.ui.lineEdit_chl.setText('')
         self.ui.lineEdit_free.setText('')
         self.ui.lineEdit_totalsul.setText('')
         self.ui.lineEdit_densi.setText('')
         self.ui.lineEdit_ph.setText('')
         self.ui.lineEdit_sulp.setText('')
         self.ui.lineEdit_alco.setText('')
         self.ui.lineEdit_qual.setText('')

