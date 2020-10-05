from PyQt5.QtWidgets import QWidget, QMessageBox, QDesktopWidget, QMainWindow, QAction, qApp, QApplication, QFileDialog
from reportmain import Ui_MainWindowrep
from addadvrecord import AddNewAdver
from addsalesrecord import AddNewSales
from addwinerecord import AddNewWine
from addretailrecord import AddNewRetail
import pandas as pd
import numpy as np
import sqlite3
import time
from PandasModel import PandasModel
import re
from sklearn.linear_model import LinearRegression
import seaborn as sns
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from math import sqrt
from scipy.io import arff
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore", "(?s).*MATPLOTLIBDATA.*", category=UserWarning)

class ReportMain(QMainWindow):
    
    def __init__(self):
        super().__init__()
        
        self.ui = Ui_MainWindowrep()
        self.ui.setupUi(self)
        
        self.addadvrecord = AddNewAdver()
        self.addsalesrecord = AddNewSales()
        self.addwinerecord = AddNewWine()
        self.addretailrecord = AddNewRetail()

        self.page_reg_index   = 0
        self.page_sales_index = 1
        self.page_quality_index = 2
        self.page_customersegment_index = 3
        self.page_regreport_index = 4
                
        self.ui.actionLog_off.triggered.connect(self.logoff)
        self.ui.actionMult_Reg.triggered.connect(self.go_regpage)
        self.ui.actionSales_Mult.triggered.connect(self.go_salespage)
        self.ui.actionQuality_Pred.triggered.connect(self.go_qualitypage)
        self.ui.actionCustomer_Seg.triggered.connect(self.go_customersegmentpage)
        self.ui.actionupload_yourdset.triggered.connect(self.go_regreportpage)
        #advert
        self.ui.pushButton_marketing_2.clicked.connect(self.loadadverttable)
        self.ui.pushButton_addnew.clicked.connect(self.addnewscreen_adver)
        self.ui.pushButton_calc.clicked.connect(self.calculate)
        self.ui.pushButton_predictsales.clicked.connect(self.predictadv)
        self.ui.Pushbutton_detail.clicked.connect(self.loadadvertdetail)
        self.ui.Pushbutton_joint_tv.clicked.connect(self.update_joint)
        self.ui.pushButton_clearadver.clicked.connect(self.clearcalculate)
        self.ui.pushButton_delete_advert.clicked.connect(self.deleteadvert)
        self.ui.pushButton_export_to_excel_advert.clicked.connect(self.exportadvertdisplayed)
        self.ui.pushButton_clearsearchadvert.clicked.connect(self.clearsearchadv)
        #sales
        self.ui.pushButton_salesdata.clicked.connect(self.loadsales)
        self.ui.pushButton_addnew_sales.clicked.connect(self.addnewscreen_sales)
        self.ui.Pushbutton_detail_sales.clicked.connect(self.loadsalesdetailproduct)
        self.ui.Pushbutton_detail_sales.clicked.connect(self.loadsalesdetailstore)
        self.ui.pushButton_pred_sales.clicked.connect(self.predsales)
        self.ui.Pushbutton_dispsales.clicked.connect(self.updateplot)
        self.ui.pushButton_export_sales.clicked.connect(self.exportsalesdisplayed)
        self.ui.pushButton_delete_sales.clicked.connect(self.deletesales)
        self.ui.pushButton_clearsearchsales.clicked.connect(self.clearsearchsales)
        #wine
        self.ui.pushButton_winedata.clicked.connect(self.loadwinedata)
        self.ui.pushButton_addnew_wine.clicked.connect(self.addnewscreen_wine)
        self.ui.Pushbutton_detail_wine.clicked.connect(self.loadwinedetail1)
        self.ui.Pushbutton_detail_wine.clicked.connect(self.loadwinedetail2)
        self.ui.pushButton_clearpre.clicked.connect(self.clearwine)
        self.ui.pushButton_dispfeature.clicked.connect(self.importance)
        self.ui.pushButton_dispclassrate.clicked.connect(self.succesrate)
        self.ui.pushButton_export_to_excel_wine.clicked.connect(self.exportwinedisplayed)
        self.ui.pushButton_delete_wine.clicked.connect(self.deletewine)
        self.ui.pushButton_clearsearchwine.clicked.connect(self.clearsearchwine)
        self.ui.pushButton_pred.clicked.connect(self.newexpresult)
        #customersegment#
        self.ui.pushButton_display_retail.clicked.connect(self.displayretail)
        self.ui.pushButton_addnew_retail.clicked.connect(self.addnewscreen_retail)
        self.ui.pushButton_display_customer.clicked.connect(self.scatterplotseg)
        self.ui.pushButton_display_customer.clicked.connect(self.overall)
        self.ui.pushButton_export_to_excel.clicked.connect(self.exportoverall)
        self.ui.pushButton_export_to_excel_in.clicked.connect(self.exportdisplayed)
        self.ui.pushButton_delete_retail.clicked.connect(self.deleteretail)
        self.ui.pushButton_clearsearchretail.clicked.connect(self.clearsearchretail)
        #regressionclassfication#
        self.ui.pushButton_choose_arff.clicked.connect(self.arffregreport)
        self.ui.pushButton_clear_class.clicked.connect(self.clearclass)
        self.ui.pushButton_clear_reg.clicked.connect(self.clearregres)
        self.ui.pushButton_choose_arff_4.clicked.connect(self.excelimport)
        
    def logoff(self):
        self.close()

    
    def go_regpage(self):
        self.ui.stackedWidget.setCurrentIndex(self.page_reg_index)
    
    def go_salespage(self):
        self.ui.stackedWidget.setCurrentIndex(self.page_sales_index)
        
    def go_qualitypage(self):
        self.ui.stackedWidget.setCurrentIndex(self.page_quality_index)
        
    def go_customersegmentpage(self):
        self.ui.stackedWidget.setCurrentIndex(self.page_customersegment_index)
        conn = sqlite3.connect('veri_tabanı.db')
        df = pd.read_sql_query("select distinct Country from OnlineRetails",conn)
        combolist = df['Country'].tolist()
        self.ui.comboBox.addItems(combolist)
        self.ui.comboBox_2.addItems(combolist)
       
    def go_regreportpage(self):
        self.ui.stackedWidget.setCurrentIndex(self.page_regreport_index)
        self.ui.lineEdit_reg_cv.setText('10')
    
    def addnewscreen_adver(self):
        self.addadvrecord.show()
        
    def addnewscreen_sales(self):
        self.addsalesrecord.show()
    
    def addnewscreen_wine(self):
        self.addwinerecord.show()
    
    def addnewscreen_retail(self):
        self.addretailrecord.show()
    
##advert##
    
    def loadadverttable(self):
        conn = sqlite3.connect('veri_tabanı.db')
        df = pd.read_sql_query("select * from Advertising", conn)
        model_s = PandasModel(df)
        self.ui.tableView_advert_display.setModel(model_s)
        self.ui.label_count_adv.setText('Count:'+ str(len(df)))
        regexrange = '^(?:0|[1-9][0-9]*)\.[0-9]+$'
        tvex1=self.ui.lineEdit_exptv_1.text()
        tvex2=self.ui.lineEdit_exptv_2.text()
        radex1=self.ui.lineEdit_exprad_1.text()
        radex2=self.ui.lineEdit_exprad_2.text()
        newex1=self.ui.lineEdit_expnew_1.text()
        newex2=self.ui.lineEdit_expnew_2.text()
        salex1=self.ui.lineEdit_expsale_1.text()
        salex2=self.ui.lineEdit_expsale_2.text()
        matchObj1 = re.search(regexrange,tvex1)
        matchObj2 = re.search(regexrange,tvex2)
        matchObj3 = re.search(regexrange,radex1)
        matchObj4 = re.search(regexrange,radex2)
        matchObj5 = re.search(regexrange,newex1)
        matchObj6 = re.search(regexrange,newex2)
        matchObj7 = re.search(regexrange,salex1)
        matchObj8 = re.search(regexrange,salex2)
        if matchObj1 and matchObj2 and tvex1 !="" and tvex2 !="":
            dfin = pd.read_sql_query(f"select * from Advertising where TV >='{tvex1}' and TV <='{tvex2}'", conn)
            modeli = PandasModel(dfin)
            self.ui.tableView_advert_display.setModel(modeli)
            self.ui.label_count_adv.setText('Count:'+ str(len(dfin)))
        if matchObj3 and matchObj4 and radex1 !="" and radex2 !="":
            dfin = pd.read_sql_query(f"select * from Advertising where radio >='{radex1}' and radio <='{radex2}'", conn)
            modeli = PandasModel(dfin)
            self.ui.tableView_advert_display.setModel(modeli)
            self.ui.label_count_adv.setText('Count:'+ str(len(dfin)))
        if matchObj5 and matchObj6 and newex1 !="" and newex2 !="":
            dfin = pd.read_sql_query(f"select * from Advertising where newspaper >='{newex1}' and newspaper <='{newex2}'", conn)
            modeli = PandasModel(dfin)
            self.ui.tableView_advert_display.setModel(modeli)
            self.ui.label_count_adv.setText('Count:'+ str(len(dfin)))
        if matchObj7 and matchObj8 and salex1 !="" and salex2 !="":
            dfin = pd.read_sql_query(f"select * from Advertising where sales >='{salex1}' and sales <='{salex2}'", conn)
            modeli = PandasModel(dfin)
            self.ui.tableView_advert_display.setModel(modeli)
            self.ui.label_count_adv.setText('Count:'+ str(len(dfin)))
            
    def exportadvertdisplayed(self):
        regexrange = '^(?:0|[1-9][0-9]*)\.[0-9]+$'
        tvex1=self.ui.lineEdit_exptv_1.text()
        tvex2=self.ui.lineEdit_exptv_2.text()
        radex1=self.ui.lineEdit_exprad_1.text()
        radex2=self.ui.lineEdit_exprad_2.text()
        newex1=self.ui.lineEdit_expnew_1.text()
        newex2=self.ui.lineEdit_expnew_2.text()
        salex1=self.ui.lineEdit_expsale_1.text()
        salex2=self.ui.lineEdit_expsale_2.text()
        matchObj1 = re.search(regexrange,tvex1)
        matchObj2 = re.search(regexrange,tvex2)
        matchObj3 = re.search(regexrange,radex1)
        matchObj4 = re.search(regexrange,radex2)
        matchObj5 = re.search(regexrange,newex1)
        matchObj6 = re.search(regexrange,newex2)
        matchObj7 = re.search(regexrange,salex1)
        matchObj8 = re.search(regexrange,salex2)
        conn = sqlite3.connect('veri_tabanı.db')
        if (tvex1 !="" and tvex2 !="" and radex1 !="" and radex2 !="" and newex1 !="" and newex2 !="" and salex1 !="" and salex2 !=""):
            dfin = pd.read_sql_query(f"select * from Advertising", conn)
            try:
                bookname = 'Filtered_Advertising'+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'All_Records'
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                dfin.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'All Advertising Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        if matchObj1 and matchObj2 and tvex1 !="" and tvex2 !="":
            dfin = pd.read_sql_query(f"select * from Advertising where TV >='{tvex1}' and TV <='{tvex2}'", conn)
            try:
                bookname = 'Filtered_'+tvex1+'-'+tvex2+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'All_Records'
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                dfin.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Advertising Tv Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        if matchObj3 and matchObj4 and radex1 !="" and radex2 !="":
            dfin = pd.read_sql_query(f"select * from Advertising  where radio >='{radex1}' and radio <='{radex2}'", conn)
            try:
                bookname = 'Filtered_'+radex1+'-'+radex2+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'All_Records'
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                dfin.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Advertising Radio Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        if matchObj5 and matchObj6 and newex1 !="" and newex2 !="":
            dfin = pd.read_sql_query(f"select * from  Advertising where newspaper >='{newex1}' and newspaper <='{newex2}'", conn)
            try:
                bookname = 'Filtered_'+newex1+'-'+newex2+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'All_Records'
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                dfin.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Advertising News Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        if matchObj7 and matchObj8 and salex1 !="" and salex2 !="":
            dfin = pd.read_sql_query(f"select * from  Advertising where sales >='{salex1}' and sales <='{salex2}'", conn)
            try:
                bookname = 'Filtered_'+salex1+'-'+salex2+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'All_Records'
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                dfin.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Advertising Sales Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
            
    def deleteadvert(self):
        conn = sqlite3.connect('veri_tabanı.db')
        df = pd.read_sql_query("select * from Advertising", conn)
        model_s = PandasModel(df)
        self.ui.tableView_advert_display.setModel(model_s)
        regexrange = '^(?:0|[1-9][0-9]*)\.[0-9]+$'
        tvex1=self.ui.lineEdit_exptv_1.text()
        tvex2=self.ui.lineEdit_exptv_2.text()
        radex1=self.ui.lineEdit_exprad_1.text()
        radex2=self.ui.lineEdit_exprad_2.text()
        newex1=self.ui.lineEdit_expnew_1.text()
        newex2=self.ui.lineEdit_expnew_2.text()
        salex1=self.ui.lineEdit_expsale_1.text()
        salex2=self.ui.lineEdit_expsale_2.text()
        matchObj1 = re.search(regexrange,tvex1)
        matchObj2 = re.search(regexrange,tvex2)
        matchObj3 = re.search(regexrange,radex1)
        matchObj4 = re.search(regexrange,radex2)
        matchObj5 = re.search(regexrange,newex1)
        matchObj6 = re.search(regexrange,newex2)
        matchObj7 = re.search(regexrange,salex1)
        matchObj8 = re.search(regexrange,salex2)
        if(tvex1 !="" and tvex2 =="" and radex1 =="" and radex2 =="" and newex1 =="" and newex2 =="" and salex1 =="" and salex2 ==""):
            QMessageBox.critical(self, "Opps!", 'You can delete a record after filtering!')
        if matchObj1 and matchObj2 and tvex1 !="" and tvex2 !="":
            dfin = pd.read_sql_query(f"select * from Advertising where TV >='{tvex1}' and TV <='{tvex2}'", conn)
            leng = "Do you want to delete " + str(len(dfin)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from Advertising where TV >='{tvex1}' and TV <='{tvex2}'")
                conn.commit()
                suc = "Tv Expense between "+tvex1+" and "+tvex2+" "+str(len(dfin))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if matchObj3 and matchObj4 and radex1 !="" and radex2 !="":
            dfin = pd.read_sql_query(f"select * from Advertising where radio >='{radex1}' and radio <='{radex2}'", conn)
            leng = "Do you want to delete " + str(len(dfin)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from Advertising where radio >='{radex1}' and radio <='{radex2}'")
                conn.commit()
                suc = "Radio Expense between "+radex1+" and "+radex2+" "+str(len(dfin))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if matchObj5 and matchObj6 and newex1 !="" and newex2 !="":
            dfin = pd.read_sql_query(f"select * from Advertising where newspaper >='{newex1}' and newspaper <='{newex2}'", conn)
            leng = "Do you want to delete " + str(len(dfin)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from Advertising where newspaper >='{newex1}' and newspaper <='{newex2}'")
                conn.commit()
                suc = "Newspaper Expense between "+newex1+" and "+newex2+" "+str(len(dfin))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if matchObj7 and matchObj8 and salex1 !="" and salex2 !="":
            dfin = pd.read_sql_query(f"select * from Advertising where sales >='{salex1}' and sales <='{salex2}'", conn)
            leng = "Do you want to delete " + str(len(dfin)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from Advertising where sales >='{salex1}' and sales <='{salex2}'")
                conn.commit()
                suc = "Sales between "+newex1+" and "+newex2+" "+str(len(dfin))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        
    
    def calculate(self):
        conn = sqlite3.connect('veri_tabanı.db')
        df = pd.read_sql_query("select * from Advertising", conn)
        df = df.iloc[:, 1:len(df)]
        X = df.drop('sales', axis=1)
        y = df[["sales"]]
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
        model = LinearRegression().fit(X_train,y_train)
        crossval = self.ui.lineEdit_crossval.text()
        fcv = int(crossval)
        mse = float('%.3f' % np.mean(-cross_val_score(model,X_train,y_train,cv= fcv, scoring= "neg_mean_squared_error")))
        mse_str = str(mse)
        rmse = float('%.3f' % np.sqrt(np.mean(-cross_val_score(model,X_train,y_train, cv= fcv, scoring= "neg_mean_squared_error"))))
        rmse_str = str(rmse)
        self.ui.lineEdit_mse.setText(mse_str)
        self.ui.lineEdit_rmse.setText(rmse_str)
    
    def predictadv(self):
        conn = sqlite3.connect('veri_tabanı.db')
        df = pd.read_sql_query("select * from Advertising", conn)
        df = df.iloc[:, 1:len(df)]
        X = df.drop('sales', axis=1)
        y = df[["sales"]]
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
        model = LinearRegression().fit(X_train,y_train)
        expensetv=self.ui.lineEdit_exptv.text()
        expenserad=self.ui.lineEdit_exprad.text()
        expensenews=self.ui.lineEdit_expnews.text()
        ftv = float(expensetv)
        frad = float(expenserad)
        fnews = float(expensenews)
        yeni_veri = [[ftv],[frad],[fnews]]
        yeni_veri = pd.DataFrame(yeni_veri).T
        deg= model.predict(yeni_veri)
        deg= float('%.3f' % deg.item(0))
        deg_str = str(deg)
        self.ui.lineEdit_forecast.setText(deg_str)
        
    def loadadvertdetail(self):
        conn = sqlite3.connect('veri_tabanı.db')
        df = pd.read_sql_query("select * from Advertising",conn)
        df = df.iloc[:,1:len(df)]
        df = df.describe()
        model = PandasModel(df)
        self.ui.tableView.setModel(model)
        X = df.drop('sales', axis=1)
        y = df[["sales"]]
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
        model = LinearRegression().fit(X_train,y_train)
        coefficients = pd.concat([pd.DataFrame(X_train.columns),pd.DataFrame(np.transpose(model.coef_)).astype('float64')], axis = 1)
        coefficients.columns = ['Attributes','Coefficients']
        coefficients.Coefficients=coefficients.Coefficients.round(4)
        model_coe = PandasModel(coefficients)
        self.ui.tableView_coef.setModel(model_coe)
        
    def update_joint(self):
        conn = sqlite3.connect('veri_tabanı.db')
        df = pd.read_sql_query("select * from Advertising",conn)
        df = df.iloc[:,1:len(df)]
        
        self.ui.MplWidget_tv.canvas.axes.clear()
        self.ui.MplWidget_tv.canvas.axes.scatter(x="TV",y="sales",data=df)
        self.ui.MplWidget_tv.canvas.axes.legend(('TV', 'Sales'),loc='upper right')
        self.ui.MplWidget_tv.canvas.axes.set_title('TV Regression Scatter')
        self.ui.MplWidget_tv.canvas.axes.set_ylabel('Sales')
        self.ui.MplWidget_tv.canvas.draw()
        
        self.ui.MplWidget_radio.canvas.axes.clear()
        self.ui.MplWidget_radio.canvas.axes.scatter(x="radio",y="sales",data=df,color= 'green')
        self.ui.MplWidget_radio.canvas.axes.legend(('radio', 'Sales'),loc='upper right')
        self.ui.MplWidget_radio.canvas.axes.set_title('Radio Regression Scatter')
        self.ui.MplWidget_radio.canvas.axes.set_ylabel('Sales')
        self.ui.MplWidget_radio.canvas.draw()
        
        self.ui.MplWidget_news.canvas.axes.clear()
        self.ui.MplWidget_news.canvas.axes.scatter(x="newspaper",y="sales",data=df,color= 'orange')
        self.ui.MplWidget_news.canvas.axes.legend(('newspaper', 'Sales'),loc='upper right')
        self.ui.MplWidget_news.canvas.axes.set_title('Newspaper Regression Scatter')
        self.ui.MplWidget_news.canvas.axes.set_ylabel('Sales')
        self.ui.MplWidget_news.canvas.draw()
    
    def clearcalculate(self):
        self.ui.lineEdit_exptv.setText('')
        self.ui.lineEdit_exprad.setText('')
        self.ui.lineEdit_expnews.setText('')
        self.ui.lineEdit_crossval.setText('')
        self.ui.lineEdit_forecast.setText('')
        self.ui.lineEdit_mse.setText('')
        self.ui.lineEdit_rmse.setText('')
    
    def clearsearchadv(self):
        self.ui.lineEdit_exptv_1.setText('')
        self.ui.lineEdit_exptv_2.setText('')
        self.ui.lineEdit_exprad_1.setText('')
        self.ui.lineEdit_exprad_2.setText('')
        self.ui.lineEdit_expnew_1.setText('')
        self.ui.lineEdit_expnew_2.setText('')
        self.ui.lineEdit_expsale_1.setText('')
        self.ui.lineEdit_expsale_2.setText('')
        
##sales##

    def loadsales(self):
        storeid = self.ui.lineEdit_store_id.text()
        productn = self.ui.lineEdit_productname.text()
        ısholiday = self.ui.lineEdit_isholiday.text()
        productprice = self.ui.lineEdit_product_price.text()
        recordyear = self.ui.lineEdit_record_year.text()
        weeklyunit = self.ui.lineEdit_weekly_unit.text()
        conn = sqlite3.connect('veri_tabanı.db')
        if(storeid == "" and productn == "" and ısholiday == "" and productprice == "" and recordyear =="" and weeklyunit == ""):
            dfin = pd.read_sql_query("select * from Stores_Sales",conn)
            model = PandasModel(dfin)
            self.ui.tableView_sales.setModel(model)
            self.ui.label_count_sales.setText('Count:'+ str(len(dfin)))
        if(storeid != "" and productn == "" and ısholiday == "" and productprice == "" and recordyear =="" and weeklyunit == ""):
            dfin = pd.read_sql_query(f"select * from Stores_Sales where Store like '{storeid}%'",conn)
            model = PandasModel(dfin)
            self.ui.tableView_sales.setModel(model)
            self.ui.label_count_sales.setText('Count:'+ str(len(dfin)))
        if(productn != "" and storeid == "" and ısholiday == "" and productprice == "" and recordyear =="" and weeklyunit == ""):
            dfin = pd.read_sql_query(f"select * from Stores_Sales where Product like '{productn}%'",conn)
            model = PandasModel(dfin)
            self.ui.tableView_sales.setModel(model)
            self.ui.label_count_sales.setText('Count:'+ str(len(dfin)))
        if(ısholiday !="" and storeid == "" and productn == "" and productprice == "" and recordyear =="" and weeklyunit == ""):
            dfin = pd.read_sql_query(f"select * from Stores_Sales where Is_Holiday like '{ısholiday}%'",conn)
            model = PandasModel(dfin)
            self.ui.tableView_sales.setModel(model)
            self.ui.label_count_sales.setText('Count:'+ str(len(dfin)))
        if(storeid != "" and productn != "" and ısholiday == "" and productprice == "" and recordyear =="" and weeklyunit == ""):
            dfin = pd.read_sql_query(f"select * from Stores_Sales where Store like '{storeid}%' and Product like '{productn}%' ",conn)
            model = PandasModel(dfin)
            self.ui.tableView_sales.setModel(model)
            self.ui.label_count_sales.setText('Count:'+ str(len(dfin)))
        if(storeid != "" and productn != "" and  ısholiday !="" and productprice == "" and recordyear =="" and weeklyunit == ""):
            dfin = pd.read_sql_query(f"select * from Stores_Sales where Store like '{storeid}%' and Product like '{productn}%' and Is_Holiday like '{ısholiday}%'",conn)
            model = PandasModel(dfin)
            self.ui.tableView_sales.setModel(model)
            self.ui.label_count_sales.setText('Count:'+ str(len(dfin)))
        if(productprice != "" and storeid == "" and productn == "" and ısholiday == "" and recordyear =="" and weeklyunit == ""):
            dfin = pd.read_sql_query(f"select * from Stores_Sales where Price like '{productprice}%'",conn)
            model = PandasModel(dfin)
            self.ui.tableView_sales.setModel(model)
            self.ui.label_count_sales.setText('Count:'+ str(len(dfin)))
        if(productprice != "" and recordyear != "" and storeid == "" and productn == "" and ısholiday == "" and weeklyunit == ""):
            dfin = pd.read_sql_query(f"select * from Stores_Sales where Price like '{productprice}%' and Date like '{recordyear}%' ",conn)
            model = PandasModel(dfin)
            self.ui.tableView_sales.setModel(model)
            self.ui.label_count_sales.setText('Count:'+ str(len(dfin)))
        if(productprice != "" and weeklyunit != "" and storeid == "" and productn == "" and ısholiday == "" and recordyear ==""):
            dfin = pd.read_sql_query(f"select * from Stores_Sales where Price like '{productprice}%' and Weekly_Units_Sold like '{weeklyunit}%'",conn)
            model = PandasModel(dfin)
            self.ui.tableView_sales.setModel(model)
            self.ui.label_count_sales.setText('Count:'+ str(len(dfin)))
        if(productprice != "" and recordyear != "" and weeklyunit != "" and storeid == "" and productn == "" and ısholiday == ""):
            dfin = pd.read_sql_query(f"select * from Stores_Sales where Price like '{productprice}%' and Date like '{recordyear}%' and Weekly_Units_Sold like '{weeklyunit}%'",conn)
            model = PandasModel(dfin)
            self.ui.tableView_sales.setModel(model)
            self.ui.label_count_sales.setText('Count:'+ str(len(dfin)))
    
    def exportsalesdisplayed(self):
        storeid = self.ui.lineEdit_store_id.text()
        productn = self.ui.lineEdit_productname.text()
        ısholiday = self.ui.lineEdit_isholiday.text()
        productprice = self.ui.lineEdit_product_price.text()
        recordyear = self.ui.lineEdit_record_year.text()
        weeklyunit = self.ui.lineEdit_weekly_unit.text()
        conn = sqlite3.connect('veri_tabanı.db')
        if(storeid == "" and productn == "" and ısholiday == "" and productprice == "" and recordyear =="" and weeklyunit == ""):
            tx_data = pd.read_sql_query("select * from Stores_Sales",conn)
            try:
                bookname = 'All_Sales_'+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'All_Records'
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                tx_data.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Sales Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        if(storeid != "" and productn == "" and ısholiday == "" and productprice == "" and recordyear =="" and weeklyunit == ""):
            tx_data = pd.read_sql_query(f"select * from Stores_Sales where Store like '{storeid}%'",conn)
            try:
                bookname = 'All_Sales_'+storeid+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'All_Records'
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                tx_data.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Sales Store Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        if(productn != "" and storeid == "" and ısholiday == "" and productprice == "" and recordyear =="" and weeklyunit == ""):
            tx_data = pd.read_sql_query(f"select * from Stores_Sales where Product like '{productn}%'",conn)
            try:
                bookname = 'All_Sales_'+productn+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'All_Records'
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                tx_data.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Sales Product Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        if(ısholiday !="" and storeid == "" and productn == "" and productprice == "" and recordyear =="" and weeklyunit == ""):
            tx_data = pd.read_sql_query(f"select * from Stores_Sales where Is_Holiday like '{ısholiday}%'",conn)
            try:
                bookname = 'All_Sales_'+ısholiday+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'All_Records'
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                tx_data.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Sales Is_Holiday Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        if(storeid != "" and productn != "" and ısholiday == "" and productprice == "" and recordyear =="" and weeklyunit == ""):
            tx_data = pd.read_sql_query(f"select * from Stores_Sales where Store like '{storeid}%' and Product like '{productn}%' ",conn)
            try:
                bookname = 'All_Sales_'+storeid+'_'+productn+time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'All_Records'
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                tx_data.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Sales Store and Product Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        if(storeid != "" and productn != "" and  ısholiday !="" and productprice == "" and recordyear =="" and weeklyunit == ""):
            tx_data = pd.read_sql_query(f"select * from Stores_Sales where Store like '{storeid}%' and Product like '{productn}%' and Is_Holiday like '{ısholiday}%'",conn)
            try:
                bookname = 'All_Sales_'+storeid+'_'+productn+time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'All_Records'
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                tx_data.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Sales Store,Product,IsHoliday Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        if(productprice != "" and storeid == "" and productn == "" and ısholiday == "" and recordyear =="" and weeklyunit == ""):
            tx_data = pd.read_sql_query(f"select * from Stores_Sales where Price like '{productprice}%'",conn)
            try:
                bookname = 'All_Sales_'+storeid+'_'+productn+time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'All_Records'
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                tx_data.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Sales Store,Product,IsHoliday Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        if(productprice != "" and recordyear != "" and storeid == "" and productn == "" and ısholiday == "" and weeklyunit == ""):
            tx_data = pd.read_sql_query(f"select * from Stores_Sales where Price like '{productprice}%' and Date like '{recordyear}%' ",conn)
            try:
                bookname = 'All_Sales_'+productprice+'_'+recordyear+time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'All_Records'
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                tx_data.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Sales Product Price,Record Year Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        if(productprice != "" and weeklyunit != "" and storeid == "" and productn == "" and ısholiday == "" and recordyear ==""):
            tx_data = pd.read_sql_query(f"select * from Stores_Sales where Price like '{productprice}%' and Weekly_Units_Sold like '{weeklyunit}%'",conn)
            try:
                bookname = 'All_Sales_'+productprice+'_'+weeklyunit+time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'All_Records'
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                tx_data.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Sales Product Price,Weekly Unit Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        if(productprice != "" and recordyear != "" and weeklyunit != "" and storeid == "" and productn == "" and ısholiday == ""):
            tx_data = pd.read_sql_query(f"select * from Stores_Sales where Price like '{productprice}%' and Date like '{recordyear}%' and Weekly_Units_Sold like '{weeklyunit}%'",conn)
            try:
                bookname = 'All_Sales_'+productprice+'_'+weeklyunit+time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'All_Records'
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                tx_data.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Sales Product Price,Record Year,Weekly Unit Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
            
    def deletesales(self):
        storeid = self.ui.lineEdit_store_id.text()
        productn = self.ui.lineEdit_productname.text()
        ısholiday = self.ui.lineEdit_isholiday.text()
        productprice = self.ui.lineEdit_product_price.text()
        recordyear = self.ui.lineEdit_record_year.text()
        weeklyunit = self.ui.lineEdit_weekly_unit.text()
        conn = sqlite3.connect('veri_tabanı.db')
        if(storeid == "" and productn == "" and ısholiday == "" and productprice == "" and recordyear =="" and weeklyunit == ""):
            QMessageBox.critical(self, "Opps!", 'You can delete a record after filtering!')
        if storeid != "":
            dfin = pd.read_sql_query(f"select * from Stores_Sales where Store like '{storeid}%'",conn)
            leng = "Do you want to delete " + str(len(dfin)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from Stores_Sales where Store like '{storeid}%'")
                conn.commit()
                suc = "Store_id: "+storeid+"_"+str(len(dfin))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if productn != "":
            dfin = pd.read_sql_query(f"select * from Stores_Sales where Product like '{productn}%'",conn)
            leng = "Do you want to delete " + str(len(dfin)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from Stores_Sales where Product like '{productn}%'")
                conn.commit()
                suc = "Store_id: "+storeid+"_"+str(len(dfin))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if ısholiday !="":
            dfin = pd.read_sql_query(f"select * from Stores_Sales where Is_Holiday like '{ısholiday}%'",conn)
            leng = "Do you want to delete " + str(len(dfin)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from Stores_Sales where Is_Holiday like '{ısholiday}%'")
                conn.commit()
                suc = "Store_id: "+storeid+"_"+str(len(dfin))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if storeid != "" and productn != "":
            dfin = pd.read_sql_query(f"select * from Stores_Sales where Store like '{storeid}%' and Product like '{productn}%' ",conn)
            leng = "Do you want to delete " + str(len(dfin)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from Stores_Sales where Store like '{storeid}%' and Product like '{productn}%'")
                conn.commit()
                suc = "Store_id: "+storeid+"_"+"Product:"+productn+"_"+str(len(dfin))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if storeid != "" and productn != "" and  ısholiday !="":
            dfin = pd.read_sql_query(f"select * from Stores_Sales where Store like '{storeid}%' and Product like '{productn}%' and Is_Holiday like '{ısholiday}%'",conn)
            leng = "Do you want to delete " + str(len(dfin)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from Stores_Sales where Store like '{storeid}%' and Product like '{productn}%' and Is_Holiday like '{ısholiday}%'")
                conn.commit()
                suc = "Store_id: "+storeid+"_"+"Product:"+productn+"_"+"IsHoliday:"+ısholiday+"_"+str(len(dfin))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if productprice != "":
            dfin = pd.read_sql_query(f"select * from Stores_Sales where Price like '{productprice}%'",conn)
            leng = "Do you want to delete " + str(len(dfin)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from Stores_Sales where Price like '{productprice}%'")
                conn.commit()
                suc = "Product Price: "+productprice+"_"+str(len(dfin))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if productprice != "" and recordyear != "":
            dfin = pd.read_sql_query(f"select * from Stores_Sales where Price like '{productprice}%' and Date like '{recordyear}%' ",conn)
            leng = "Do you want to delete " + str(len(dfin)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from Stores_Sales where Price like '{productprice}%' and Date like '{recordyear}%'")
                conn.commit()
                suc = "Product Price: "+productprice+"_"+"Record Year:"+recordyear+"_"+str(len(dfin))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if productprice != "" and weeklyunit != "":
            dfin = pd.read_sql_query(f"select * from Stores_Sales where Price like '{productprice}%' and Weekly_Units_Sold like '{weeklyunit}%'",conn)
            leng = "Do you want to delete " + str(len(dfin)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from Stores_Sales where Price like '{productprice}%' and Weekly_Units_Sold like '{weeklyunit}%'")
                conn.commit()
                suc = "Product Price: "+productprice+"_"+"Weekly Unit:"+weeklyunit+"_"+str(len(dfin))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if productprice != "" and recordyear != "" and weeklyunit != "":
            dfin = pd.read_sql_query(f"select * from Stores_Sales where Price like '{productprice}%' and Date like '{recordyear}%' and Weekly_Units_Sold like '{weeklyunit}%'",conn)
            leng = "Do you want to delete " + str(len(dfin)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from Stores_Sales where Price like '{productprice}%' and Date like '{recordyear}%' and Weekly_Units_Sold like '{weeklyunit}%'")
                conn.commit()
                suc = "Product Price: "+productprice+"_"+"Record Year: "+recordyear+"_"+"Weekly Unit:"+weeklyunit+"_"+str(len(dfin))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
    
    def clearsearchsales(self):
        self.ui.lineEdit_store_id.setText('')
        self.ui.lineEdit_productname.setText('')
        self.ui.lineEdit_isholiday.setText('')
        self.ui.lineEdit_product_price.setText('')
        self.ui.lineEdit_record_year.setText('')
        self.ui.lineEdit_weekly_unit.setText('')
        
    
    def loadsalesdetailproduct(self):
        conn = sqlite3.connect('veri_tabanı.db')
        df = pd.read_sql_query("select * from Stores_Sales",conn)
        df = df.groupby('Product')['Weekly_Units_Sold'].describe().T
        model = PandasModel(df)
        self.ui.tableView_sales_detail_product.setModel(model)
        
    def loadsalesdetailstore(self):
        conn = sqlite3.connect('veri_tabanı.db')
        df = pd.read_sql_query("select * from Stores_Sales",conn)
        df = df.groupby('Store')['Weekly_Units_Sold'].describe().T
        model = PandasModel(df)
        self.ui.tableView_sales_detail_store.setModel(model)
    
    def succesrate(self):
        conn = sqlite3.connect('veri_tabanı.db')
        df = pd.read_sql_query("select * from Wine",conn)
        y=df["quality"]
        X=df.drop(["quality"],axis=1)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
        cvs = int(self.ui.lineEdit_cvclass.text())
        ##cart
        cart_model = DecisionTreeClassifier().fit(X_train,y_train)
        cvc = float('%.3f' % cross_val_score(cart_model,X_test,y_test,cv=cvs).mean())*100
        st_cvc = str(round(cvc,2))
        self.ui.lineEdit_desctree.setText(st_cvc)
        ##lojistic
        loj_model = LogisticRegression(solver="liblinear").fit(X_train,y_train)
        cvl = float('%.3f' % cross_val_score(loj_model,X_test,y_test,cv=cvs).mean())*100
        st_cvl = str(round(cvl,2))
        self.ui.lineEdit_lojreg.setText(st_cvl)
        ##Kneighbours
        knn_model = KNeighborsClassifier(n_neighbors = 1).fit(X_train, y_train)
        cvk = float('%.3f' % cross_val_score(knn_model,X_test,y_test,cv=cvs).mean())*100
        st_cvk = str(round(cvk,2))
        self.ui.lineEdit_kneigh.setText(st_cvk)
        ##random forest
        rfor_model = RandomForestClassifier(max_features = 11,min_samples_split =5,n_estimators=10).fit(X_train,y_train)
        rfor_pred = rfor_model.predict(X_test)
        rfor_acc = float('%.3f' % accuracy_score(y_test,rfor_pred))*100
        st_rfor_acc = str(round(rfor_acc,2))
        self.ui.lineEdit_random.setText(st_rfor_acc)
        ##LDA
        lda_model = LDA().fit(X_train,y_train)
        cvl = float('%.3f' % cross_val_score(lda_model,X_test,y_test,cv=cvs).mean())*100
        st_cvl = str(round(cvl,2))
        self.ui.lineEdit_mlp.setText(st_cvl)
    
    def predsales(self):
        conn = sqlite3.connect('veri_tabanı.db')
        df = pd.read_sql_query("select * from Stores_Sales",conn)
        y= df['Weekly_Units_Sold']
        X_ = df.drop(['Store','Product','Is_Holiday','Weekly_Units_Sold','Date','BasePrice'],axis=1).astype('float64')
        X = pd.concat([X_,pd.get_dummies(df[['Product','Is_Holiday','Store']])],axis=1)
        # X = [[price],[Product_Chocolate],[Product_FruitJuice],[Product_IceCream],[Is_Holiday_false],[Is_Holiday_true],[Store_1],[Store_2],[Store_3]]
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
        loj_model = ElasticNet().fit(X_train,y_train)
        price = float(self.ui.lineEdit_price.text())
        #is_holiday_true
        if self.ui.radioButton_choc.isChecked() and self.ui.radioButton_store1.isChecked() and self.ui.radioButton_true.isChecked():
            new_exp = [[price],[1],[0],[0],[0],[1],[1],[0],[0]]
        if self.ui.radioButton_fru.isChecked() and self.ui.radioButton_store1.isChecked() and self.ui.radioButton_true.isChecked():
            new_exp = [[price],[0],[1],[0],[0],[1],[1],[0],[0]]
        if self.ui.radioButton_ice.isChecked() and self.ui.radioButton_store1.isChecked() and self.ui.radioButton_true.isChecked():
            new_exp = [[price],[0],[0],[1],[0],[1],[1],[0],[0]]
        if self.ui.radioButton_choc.isChecked() and self.ui.radioButton_store2.isChecked() and self.ui.radioButton_true.isChecked():
            new_exp = [[price],[1],[0],[0],[0],[1],[0],[1],[0]]
        if self.ui.radioButton_fru.isChecked() and self.ui.radioButton_store2.isChecked() and self.ui.radioButton_true.isChecked():
            new_exp = [[price],[0],[1],[0],[0],[1],[0],[1],[0]]
        if self.ui.radioButton_ice.isChecked() and self.ui.radioButton_store2.isChecked() and self.ui.radioButton_true.isChecked():
            new_exp = [[price],[0],[0],[1],[0],[1],[0],[1],[0]]    
        if self.ui.radioButton_choc.isChecked() and self.ui.radioButton_store3.isChecked() and self.ui.radioButton_true.isChecked():
            new_exp = [[price],[1],[0],[0],[0],[1],[0],[0],[1]]
        if self.ui.radioButton_fru.isChecked() and self.ui.radioButton_store3.isChecked() and self.ui.radioButton_true.isChecked():
            new_exp = [[price],[0],[1],[0],[0],[1],[0],[0],[1]]
        if self.ui.radioButton_ice.isChecked() and self.ui.radioButton_store3.isChecked() and self.ui.radioButton_true.isChecked():
            new_exp = [[price],[0],[0],[1],[0],[1],[0],[0],[1]]
        ##is_holiday_false
        if self.ui.radioButton_choc.isChecked() and self.ui.radioButton_store1.isChecked() and self.ui.radioButton_false.isChecked():
            new_exp = [[price],[1],[0],[0],[1],[0],[1],[0],[0]]
        if self.ui.radioButton_fru.isChecked() and self.ui.radioButton_store1.isChecked() and self.ui.radioButton_false.isChecked():
            new_exp = [[price],[0],[1],[0],[1],[0],[1],[0],[0]]
        if self.ui.radioButton_ice.isChecked() and self.ui.radioButton_store1.isChecked() and self.ui.radioButton_false.isChecked():
            new_exp = [[price],[0],[0],[1],[1],[0],[1],[0],[0]]
        if self.ui.radioButton_choc.isChecked() and self.ui.radioButton_store2.isChecked() and self.ui.radioButton_false.isChecked():
            new_exp = [[price],[1],[0],[0],[1],[0],[0],[1],[0]]
        if self.ui.radioButton_fru.isChecked() and self.ui.radioButton_store2.isChecked() and self.ui.radioButton_false.isChecked():
            new_exp = [[price],[0],[1],[0],[1],[0],[0],[1],[0]]
        if self.ui.radioButton_ice.isChecked() and self.ui.radioButton_store2.isChecked() and self.ui.radioButton_false.isChecked():
            new_exp = [[price],[0],[0],[1],[1],[0],[0],[1],[0]]
        if self.ui.radioButton_choc.isChecked() and self.ui.radioButton_store3.isChecked() and self.ui.radioButton_false.isChecked():
            new_exp = [[price],[1],[0],[0],[1],[0],[0],[0],[1]]
        if self.ui.radioButton_fru.isChecked() and self.ui.radioButton_store3.isChecked() and self.ui.radioButton_false.isChecked():
            new_exp = [[price],[0],[1],[0],[1],[0],[0],[0],[1]]
        if self.ui.radioButton_ice.isChecked() and self.ui.radioButton_store3.isChecked() and self.ui.radioButton_false.isChecked():
            new_exp = [[price],[0],[0],[1],[1],[0],[0],[0],[1]]
        new_exp = pd.DataFrame(new_exp).T
        pre = loj_model.predict(new_exp)
        pre = float('%.3f' % pre.item(0))
        st_pre = str(pre)
        self.ui.lineEdit_weeklyunit.setText(st_pre)
        weeklysales = str(float('%.3f' %(float(price) * float(pre))))
        self.ui.lineEdit_weeklyunit_2.setText(weeklysales)

    
    def updateplot(self):    
        conn = sqlite3.connect('veri_tabanı.db')
        df = pd.read_sql_query("select * from Stores_Sales",conn)
        df['Date'] = pd.to_datetime(df['Date'])
        df['weekly_sales'] = df['Price'] * df['Weekly_Units_Sold']
        
        sns.set(style = "ticks")
        c = '#c553a0'
        cdf = ECDF(df['Weekly_Units_Sold'])

        self.ui.MplWidget_2.canvas.axes.clear()
        self.ui.MplWidget_2.canvas.axes.plot(cdf.x, cdf.y,'r.-', label = "statmodels", color = c)
        self.ui.MplWidget_2.canvas.axes.legend(('Weekly Units Sold', 'ECDF'),loc='upper right')
        self.ui.MplWidget_2.canvas.draw()
        
        sns.set(style = "ticks")
        c = '#4dc0d4'
        cdfs = ECDF(df['weekly_sales'])
        
        self.ui.MplWidget_3.canvas.axes.clear()
        self.ui.MplWidget_3.canvas.axes.plot(cdfs.x, cdfs.y,'r.-', label = "statmodels", color = c)
        self.ui.MplWidget_3.canvas.axes.legend(('Weekly sales', 'ECDF'),loc='upper right')
        self.ui.MplWidget_3.canvas.draw()
        
        
        dfp= df.groupby('Product')['weekly_sales'].sum().reset_index()
        sizes = dfp['weekly_sales'].tolist()
        labels = dfp['Product'].tolist()
        colors = ['#cb69eb','#4dc0d4','#c553a0','#77e0fe']
        self.ui.MplWidget_sales.canvas.axes.clear()
        self.ui.MplWidget_sales.canvas.axes.pie(sizes,labels=labels,autopct='%1.0f%%',radius=.9,startangle = 90,colors=colors,explode=(0.05, 0.05, 0.05),shadow=True)
        self.ui.MplWidget_sales.canvas.axes.axis('equal')
        self.ui.MplWidget_sales.canvas.axes.set_title('Product - Weekly Sales Pie Chart',weight='bold')
        self.ui.MplWidget_sales.canvas.draw()          
        dfs= df.groupby('Store')['weekly_sales'].sum().reset_index()
        size = dfs['weekly_sales'].tolist()
        label = dfs['Store'].tolist()   
        colors = ['#91f581','#66b3ff','#ff99e6','#ffcc99']
        self.ui.MplWidget_sales_2.canvas.axes.clear()
        self.ui.MplWidget_sales_2.canvas.axes.pie(size,labels=label,autopct='%1.0f%%',radius=.9,startangle = 90,colors=colors,explode=(0.05, 0.05, 0.05),shadow=True)
        self.ui.MplWidget_sales_2.canvas.axes.axis('equal')
        self.ui.MplWidget_sales_2.canvas.axes.set_title('Store - Weekly Sales Pie Chart',weight='bold')
        self.ui.MplWidget_sales_2.canvas.draw()  
        
        
          
#wine
    def loadwinedata(self):
        conn = sqlite3.connect('veri_tabanı.db')
        df = pd.read_sql_query("select * from Wine",conn)
        model = PandasModel(df)
        self.ui.tableView_wine.setModel(model)
        self.ui.label_count_wine.setText('Count:'+ str(len(df)))
        regexrange = '^(?:\d{1,6})?(?:\.\d{1,6})?$'
        fix1 = self.ui.lineEdit_w1.text()
        fix2 = self.ui.lineEdit_w2.text()
        vola1 = self.ui.lineEdit_w3.text()
        vola2 = self.ui.lineEdit_w4.text()
        cit1= self.ui.lineEdit_w5.text()
        cit2= self.ui.lineEdit_w6.text()
        res1= self.ui.lineEdit_w7.text()
        res2=self.ui.lineEdit_w8.text()
        cho1=self.ui.lineEdit_w9.text()
        cho2=self.ui.lineEdit_w10.text()
        alc1=self.ui.lineEdit_w11.text()
        alc2=self.ui.lineEdit_w12.text()
        free1=self.ui.lineEdit_w13.text()
        free2=self.ui.lineEdit_w14.text()
        total1=self.ui.lineEdit_w15.text()
        total2=self.ui.lineEdit_w16.text()
        densi1=self.ui.lineEdit_w17.text()
        densi2 = self.ui.lineEdit_w18.text()
        ph1= self.ui.lineEdit_w19.text()
        ph2= self.ui.lineEdit_w20.text()
        sulp1=self.ui.lineEdit_w21.text()
        sulp2=self.ui.lineEdit_w22.text()
        qual1=self.ui.lineEdit_w23.text()
        qual2=self.ui.lineEdit_w24.text()
        mobfix1 = re.search(regexrange,fix1)
        mobfix2 = re.search(regexrange,fix2)
        mobvola1 = re.search(regexrange,vola1) 
        mobvola2 = re.search(regexrange,vola2)
        mobcit1 = re.search(regexrange,cit1)
        mobcit2 = re.search(regexrange,cit2)
        mobres1 = re.search(regexrange,res1)
        mobres2 = re.search(regexrange,res2)
        mobcho1 = re.search(regexrange,cho1)
        mobcho2 = re.search(regexrange,cho2)
        mobalc1 = re.search(regexrange,alc1)
        mobalc2 = re.search(regexrange,alc2)
        mobfree1 = re.search(regexrange,free1)
        mobfree2 = re.search(regexrange,free2)
        mobtotal1 = re.search(regexrange,total1)
        mobtotal2 = re.search(regexrange,total2)
        mobdensi1 = re.search(regexrange,densi1)
        mobdensi2 = re.search(regexrange,densi2)
        mobph1 = re.search(regexrange,ph1)
        mobph2 = re.search(regexrange,ph2)
        mobsulp1 = re.search(regexrange,sulp1)
        mobsulp2 = re.search(regexrange,sulp2)
        mobqual1 = re.search(regexrange,qual1)
        mobqual2 = re.search(regexrange,qual2)
        if mobfix1 and mobfix2 and fix1 !="" and fix2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where fixedacidity >='{fix1}' and fixedacidity <='{fix2}'", conn)
            modeli = PandasModel(dfin)
            self.ui.tableView_wine.setModel(modeli)
            self.ui.label_count_wine.setText('Count:'+ str(len(dfin)))
        if mobvola1 and mobvola2 and vola1 !="" and vola2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where volatileacidity >='{vola1}' and volatileacidity <='{vola2}'", conn)
            modeli = PandasModel(dfin)
            self.ui.tableView_wine.setModel(modeli)
            self.ui.label_count_wine.setText('Count:'+ str(len(dfin)))
        if mobcit1 and mobcit2 and cit1 !="" and cit2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where citricacid >='{cit1}' and citricacid <='{cit2}'", conn)
            modeli = PandasModel(dfin)
            self.ui.tableView_wine.setModel(modeli)
            self.ui.label_count_wine.setText('Count:'+ str(len(dfin)))
        if mobres1 and mobres2 and res1 !="" and res2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where residualsugar >='{res1}' and residualsugar <='{res2}'", conn)
            modeli = PandasModel(dfin)
            self.ui.tableView_wine.setModel(modeli)
            self.ui.label_count_wine.setText('Count:'+ str(len(dfin)))
        if mobcho1 and mobcho2 and cho1 !="" and cho2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where chlorides >='{cho1}' and chlorides <='{cho2}'", conn)
            modeli = PandasModel(dfin)
            self.ui.tableView_wine.setModel(modeli)
            self.ui.label_count_wine.setText('Count:'+ str(len(dfin)))
        if mobfree1 and mobfree2 and free1 !="" and free2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where freesulfurdioxide >='{free1}' and freesulfurdioxide <='{free2}'", conn)
            modeli = PandasModel(dfin)
            self.ui.tableView_wine.setModel(modeli)
            self.ui.label_count_wine.setText('Count:'+ str(len(dfin)))
        if mobtotal1 and mobtotal2 and  total1 !="" and total2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where totalsulfurdioxide >='{total1}' and totalsulfurdioxide <='{total2}'", conn)
            modeli = PandasModel(dfin)
            self.ui.tableView_wine.setModel(modeli)
            self.ui.label_count_wine.setText('Count:'+ str(len(dfin)))
        if mobdensi1 and mobdensi2 and  densi1 !="" and densi2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where density >='{densi1}' and density <='{densi2}'", conn)
            modeli = PandasModel(dfin)
            self.ui.tableView_wine.setModel(modeli)
            self.ui.label_count_wine.setText('Count:'+ str(len(dfin)))
        if mobph1 and mobph2 and  ph1 !="" and ph2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where pH >='{ph1}' and pH <='{ph2}'", conn)
            modeli = PandasModel(dfin)
            self.ui.tableView_wine.setModel(modeli)
            self.ui.label_count_wine.setText('Count:'+ str(len(dfin)))
        if mobsulp1 and mobsulp2 and  sulp1 !="" and sulp2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where sulphates >='{sulp1}' and sulphates <='{sulp2}'", conn)
            modeli = PandasModel(dfin)
            self.ui.tableView_wine.setModel(modeli)
            self.ui.label_count_wine.setText('Count:'+ str(len(dfin)))
        if mobalc1 and mobalc2 and alc1 !="" and alc2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where alcohol >='{alc1}' and alcohol <='{alc2}'", conn)
            modeli = PandasModel(dfin)
            self.ui.tableView_wine.setModel(modeli)
            self.ui.label_count_wine.setText('Count:'+ str(len(dfin)))
        if mobqual1 and mobqual2 and  qual1 !="" and qual2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where quality >='{qual1}' and quality <='{qual2}'", conn)
            modeli = PandasModel(dfin)
            self.ui.tableView_wine.setModel(modeli)
            self.ui.label_count_wine.setText('Count:'+ str(len(dfin)))
            
    def exportwinedisplayed(self):
        regexrange = '^(?:\d{1,6})?(?:\.\d{1,6})?$'
        fix1 = self.ui.lineEdit_w1.text()
        fix2 = self.ui.lineEdit_w2.text()
        vola1 = self.ui.lineEdit_w3.text()
        vola2 = self.ui.lineEdit_w4.text()
        cit1= self.ui.lineEdit_w5.text()
        cit2= self.ui.lineEdit_w6.text()
        res1= self.ui.lineEdit_w7.text()
        res2=self.ui.lineEdit_w8.text()
        cho1=self.ui.lineEdit_w9.text()
        cho2=self.ui.lineEdit_w10.text()
        alc1=self.ui.lineEdit_w11.text()
        alc2=self.ui.lineEdit_w12.text()
        free1=self.ui.lineEdit_w13.text()
        free2=self.ui.lineEdit_w14.text()
        total1=self.ui.lineEdit_w15.text()
        total2=self.ui.lineEdit_w16.text()
        densi1=self.ui.lineEdit_w17.text()
        densi2 = self.ui.lineEdit_w18.text()
        ph1= self.ui.lineEdit_w19.text()
        ph2= self.ui.lineEdit_w20.text()
        sulp1=self.ui.lineEdit_w21.text()
        sulp2=self.ui.lineEdit_w22.text()
        qual1=self.ui.lineEdit_w23.text()
        qual2=self.ui.lineEdit_w24.text()
        mobfix1 = re.search(regexrange,fix1)
        mobfix2 = re.search(regexrange,fix2)
        mobvola1 = re.search(regexrange,vola1) 
        mobvola2 = re.search(regexrange,vola2)
        mobcit1 = re.search(regexrange,cit1)
        mobcit2 = re.search(regexrange,cit2)
        mobres1 = re.search(regexrange,res1)
        mobres2 = re.search(regexrange,res2)
        mobcho1 = re.search(regexrange,cho1)
        mobcho2 = re.search(regexrange,cho2)
        mobalc1 = re.search(regexrange,alc1)
        mobalc2 = re.search(regexrange,alc2)
        mobfree1 = re.search(regexrange,free1)
        mobfree2 = re.search(regexrange,free2)
        mobtotal1 = re.search(regexrange,total1)
        mobtotal2 = re.search(regexrange,total2)
        mobdensi1 = re.search(regexrange,densi1)
        mobdensi2 = re.search(regexrange,densi2)
        mobph1 = re.search(regexrange,ph1)
        mobph2 = re.search(regexrange,ph2)
        mobsulp1 = re.search(regexrange,sulp1)
        mobsulp2 = re.search(regexrange,sulp2)
        mobqual1 = re.search(regexrange,qual1)
        mobqual2 = re.search(regexrange,qual2)
        conn = sqlite3.connect('veri_tabanı.db')
        if (fix1 =="" and fix2 =="" and vola1=="" and vola2=="" and cit1=="" and cit2=="" and
        res1=="" and res2=="" and cho1=="" and cho2=="" and alc1=="" and alc2=="" and free1=="" and 
        free2=="" and total1=="" and total2=="" and densi1=="" and densi2==""):
            dfin = pd.read_sql_query(f"select * from Wine", conn)
            try:
                bookname = 'Filtered_Wine'+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'All_Records'
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                dfin.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'All Wine Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        if mobfix1 and mobfix2 and fix1 !="" and fix2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where fixedacidity >='{fix1}' and fixedacidity <='{fix2}'", conn)
            try:
                bookname = 'Filtered_'+fix1+'-'+fix2+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'fixedacidity'
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                dfin.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Wine fixedacidity filtered Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        if mobvola1 and mobvola2 and vola1 !="" and vola2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where volatileacidity >='{vola1}' and volatileacidity <='{vola2}'", conn)
            try:
                bookname = 'Filtered_'+vola1+'-'+vola2+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'volatileacidity'
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                dfin.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Wine volatileacidity filtered Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        if mobcit1 and mobcit2 and cit1 !="" and cit2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where citricacid >='{cit1}' and citricacid <='{cit2}'", conn)
            try:
                bookname = 'Filtered_'+cit1+'-'+cit2+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'citricacid'
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                dfin.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Wine citricacid filtered Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        if mobres1 and mobres2 and res1 !="" and res2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where residualsugar >='{res1}' and residualsugar <='{res2}'", conn)
            try:
                bookname = 'Filtered_'+res1+'-'+res2+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'residualsugar'
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                dfin.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Wine residualsugar filtered Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        if mobcho1 and mobcho2 and cho1 !="" and cho2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where chlorides >='{cho1}' and chlorides <='{cho2}'", conn)
            try:
                bookname = 'Filtered_'+cho1+'-'+cho2+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'chlorides'
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                dfin.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Wine chlorides filtered Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        if mobfree1 and mobfree2 and free1 !="" and free2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where freesulfurdioxide >='{free1}' and freesulfurdioxide <='{free2}'", conn)
            try:
                bookname = 'Filtered_'+free1+'-'+free2+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'freesulfurdioxide'
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                dfin.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Wine freesulfurdioxide filtered Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        
        if mobtotal1 and mobtotal2 and  total1 !="" and total2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where totalsulfurdioxide >='{total1}' and totalsulfurdioxide <='{total2}'", conn)
            try:
                bookname = 'Filtered_'+total1+'-'+total2+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'totalsulfurdioxide'
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                dfin.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Wine totalsulfurdioxide filtered Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        
        if mobdensi1 and mobdensi2 and  densi1 !="" and densi2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where density >='{densi1}' and density <='{densi2}'", conn)
            try:
                bookname = 'Filtered_'+densi1+'-'+densi2+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'density'
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                dfin.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Wine density filtered Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        if mobph1 and mobph2 and  ph1 !="" and ph2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where pH >='{ph1}' and pH <='{ph2}'", conn)
            try:
                bookname = 'Filtered_'+ph1+'-'+ph2+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'pH'
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                dfin.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Wine pH filtered Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')  
        if mobsulp1 and mobsulp2 and  sulp1 !="" and sulp2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where sulphates >='{sulp1}' and sulphates <='{sulp2}'", conn)
            try:
                bookname = 'Filtered_'+sulp1+'-'+sulp2+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'sulphates'
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                dfin.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Wine sulphates filtered Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        
        if mobalc1 and mobalc2 and alc1 !="" and alc2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where alcohol >='{alc1}' and alcohol <='{alc2}'", conn)
            try:
                bookname = 'Filtered_'+alc1+'-'+alc2+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'alcohol'
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                dfin.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Wine alcohol filtered Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        if mobqual1 and mobqual2 and  qual1 !="" and qual2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where quality >='{qual1}' and quality <='{qual2}'", conn)
            try:
                bookname = 'Filtered_'+qual1+'-'+qual2+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'quality'
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                dfin.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Wine quality Filetered Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
                
    def deletewine(self):
        regexrange = '^(?:\d{1,6})?(?:\.\d{1,6})?$'
        fix1 = self.ui.lineEdit_w1.text()
        fix2 = self.ui.lineEdit_w2.text()
        vola1 = self.ui.lineEdit_w3.text()
        vola2 = self.ui.lineEdit_w4.text()
        cit1= self.ui.lineEdit_w5.text()
        cit2= self.ui.lineEdit_w6.text()
        res1= self.ui.lineEdit_w7.text()
        res2=self.ui.lineEdit_w8.text()
        cho1=self.ui.lineEdit_w9.text()
        cho2=self.ui.lineEdit_w10.text()
        alc1=self.ui.lineEdit_w11.text()
        alc2=self.ui.lineEdit_w12.text()
        free1=self.ui.lineEdit_w13.text()
        free2=self.ui.lineEdit_w14.text()
        total1=self.ui.lineEdit_w15.text()
        total2=self.ui.lineEdit_w16.text()
        densi1=self.ui.lineEdit_w17.text()
        densi2 = self.ui.lineEdit_w18.text()
        ph1= self.ui.lineEdit_w19.text()
        ph2= self.ui.lineEdit_w20.text()
        sulp1=self.ui.lineEdit_w21.text()
        sulp2=self.ui.lineEdit_w22.text()
        qual1=self.ui.lineEdit_w23.text()
        qual2=self.ui.lineEdit_w24.text()
        mobfix1 = re.search(regexrange,fix1)
        mobfix2 = re.search(regexrange,fix2)
        mobvola1 = re.search(regexrange,vola1) 
        mobvola2 = re.search(regexrange,vola2)
        mobcit1 = re.search(regexrange,cit1)
        mobcit2 = re.search(regexrange,cit2)
        mobres1 = re.search(regexrange,res1)
        mobres2 = re.search(regexrange,res2)
        mobcho1 = re.search(regexrange,cho1)
        mobcho2 = re.search(regexrange,cho2)
        mobalc1 = re.search(regexrange,alc1)
        mobalc2 = re.search(regexrange,alc2)
        mobfree1 = re.search(regexrange,free1)
        mobfree2 = re.search(regexrange,free2)
        mobtotal1 = re.search(regexrange,total1)
        mobtotal2 = re.search(regexrange,total2)
        mobdensi1 = re.search(regexrange,densi1)
        mobdensi2 = re.search(regexrange,densi2)
        mobph1 = re.search(regexrange,ph1)
        mobph2 = re.search(regexrange,ph2)
        mobsulp1 = re.search(regexrange,sulp1)
        mobsulp2 = re.search(regexrange,sulp2)
        mobqual1 = re.search(regexrange,qual1)
        mobqual2 = re.search(regexrange,qual2)
        conn = sqlite3.connect('veri_tabanı.db')
        if (fix1 =="" and fix2 =="" and vola1=="" and vola2=="" and cit1=="" and cit2=="" and res1=="" and res2=="" and cho1=="" and cho2=="" and alc1=="" and alc2=="" and free1=="" and free2=="" and total1=="" and total2=="" and densi1=="" and densi2==""):
            QMessageBox.critical(self, "Opps!", 'You can delete a record after filtering!')
        if mobfix1 and mobfix2 and fix1 !="" and fix2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where fixedacidity >='{fix1}' and fixedacidity <='{fix2}'", conn)
            leng = "Do you want to delete " + str(len(dfin)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from Wine where fixedacidity >='{fix1}' and fixedacidity <='{fix2}'")
                conn.commit()
                suc = "fixedacidity between "+fix1+" and "+fix2+" "+str(len(dfin))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if mobvola1 and mobvola2 and vola1 !="" and vola2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where volatileacidity >='{vola1}' and volatileacidity <='{vola2}'", conn)
            leng = "Do you want to delete " + str(len(dfin)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from Wine volatileacidity >='{vola1}' and volatileacidity <='{vola2}'")
                conn.commit()
                suc = "fixedacidity between "+vola1+" and "+vola2+" "+str(len(dfin))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if mobcit1 and mobcit2 and cit1 !="" and cit2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where citricacid >='{cit1}' and citricacid <='{cit2}'", conn)
            leng = "Do you want to delete " + str(len(dfin)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from Wine where citricacid >='{cit1}' and citricacid <='{cit2}'")
                conn.commit()
                suc = "fixedacidity between "+cit1+" and "+cit2+" "+str(len(dfin))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if mobres1 and mobres2 and res1 !="" and res2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where residualsugar >='{res1}' and residualsugar <='{res2}'", conn)
            leng = "Do you want to delete " + str(len(dfin)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from Wine where residualsugar >='{res1}' and residualsugar <='{res2}'")
                conn.commit()
                suc = "fixedacidity between "+res1+" and "+res2+" "+str(len(dfin))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if mobcho1 and mobcho2 and cho1 !="" and cho2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where chlorides >='{cho1}' and chlorides <='{cho2}'", conn)
            leng = "Do you want to delete " + str(len(dfin)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from Wine where chlorides >='{cho1}' and chlorides <='{cho2}'")
                conn.commit()
                suc = "fixedacidity between "+cho1+" and "+cho2+" "+str(len(dfin))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if mobfree1 and mobfree2 and free1 !="" and free2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where freesulfurdioxide >='{free1}' and freesulfurdioxide <='{free2}'", conn)
            leng = "Do you want to delete " + str(len(dfin)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from Wine where freesulfurdioxide >='{free1}' and freesulfurdioxide <='{free2}'")
                conn.commit()
                suc = "fixedacidity between "+free1+" and "+free2+" "+str(len(dfin))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if mobtotal1 and mobtotal2 and  total1 !="" and total2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where totalsulfurdioxide >='{total1}' and totalsulfurdioxide <='{total2}'", conn)
            leng = "Do you want to delete " + str(len(dfin)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from Wine where totalsulfurdioxide >='{total1}' and totalsulfurdioxide <='{total2}'")
                conn.commit()
                suc = "fixedacidity between "+total1+" and "+total2+" "+str(len(dfin))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if mobdensi1 and mobdensi2 and  densi1 !="" and densi2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where density >='{densi1}' and density <='{densi2}'", conn)
            leng = "Do you want to delete " + str(len(dfin)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from Wine where density >='{densi1}' and density <='{densi2}'")
                conn.commit()
                suc = "fixedacidity between "+total1+" and "+total2+" "+str(len(dfin))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if mobph1 and mobph2 and  ph1 !="" and ph2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where pH >='{ph1}' and pH <='{ph2}'", conn)
            leng = "Do you want to delete " + str(len(dfin)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from Wine where pH >='{ph1}' and pH <='{ph2}'")
                conn.commit()
                suc = "fixedacidity between "+ph1+" and "+ph2+" "+str(len(dfin))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if mobsulp1 and mobsulp2 and  sulp1 !="" and sulp2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where sulphates >='{sulp1}' and sulphates <='{sulp2}'", conn)
            leng = "Do you want to delete " + str(len(dfin)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from Wine where sulphates >='{sulp1}' and sulphates <='{sulp2}'")
                conn.commit()
                suc = "fixedacidity between "+sulp1+" and "+sulp2+" "+str(len(dfin))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if mobalc1 and mobalc2 and alc1 !="" and alc2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where alcohol >='{alc1}' and alcohol <='{alc2}'", conn)
            leng = "Do you want to delete " + str(len(dfin)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from Wine where alcohol >='{alc1}' and alcohol <='{alc2}'")
                conn.commit()
                suc = "fixedacidity between "+alc1+" and "+alc2+" "+str(len(dfin))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if mobqual1 and mobqual2 and  qual1 !="" and qual2 !="":
            dfin = pd.read_sql_query(f"select * from Wine where quality >='{qual1}' and quality <='{qual2}'", conn)
            leng = "Do you want to delete " + str(len(dfin)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from Wine where quality >='{qual1}' and quality <='{qual2}'", conn)
                conn.commit()
                suc = "fixedacidity between "+qual1+" and "+qual2+" "+str(len(dfin))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
            
    def clearsearchwine(self):
        self.ui.lineEdit_w1.setText('')
        self.ui.lineEdit_w2.setText('')
        self.ui.lineEdit_w3.setText('')
        self.ui.lineEdit_w4.setText('')
        self.ui.lineEdit_w5.setText('')
        self.ui.lineEdit_w6.setText('')
        self.ui.lineEdit_w7.setText('')
        self.ui.lineEdit_w8.setText('')
        self.ui.lineEdit_w9.setText('')
        self.ui.lineEdit_w10.setText('')
        self.ui.lineEdit_w11.setText('')
        self.ui.lineEdit_w12.setText('')
        self.ui.lineEdit_w13.setText('')
        self.ui.lineEdit_w14.setText('')
        self.ui.lineEdit_w15.setText('')
        self.ui.lineEdit_w16.setText('')
        self.ui.lineEdit_w17.setText('')
        self.ui.lineEdit_w18.setText('')
        self.ui.lineEdit_w19.setText('')
        self.ui.lineEdit_w20.setText('')
        self.ui.lineEdit_w21.setText('')
        self.ui.lineEdit_w22.setText('')
        self.ui.lineEdit_w23.setText('')
        self.ui.lineEdit_w24.setText('')

    def loadwinedetail1(self):
        conn = sqlite3.connect('veri_tabanı.db')
        df = pd.read_sql_query("select * from Wine",conn)
        df = df.groupby('quality')['alcohol'].describe()
        model = PandasModel(df)
        self.ui.tableView_wine_detail.setModel(model)
    
    def loadwinedetail2(self):
        conn = sqlite3.connect('veri_tabanı.db')
        df = pd.read_sql_query("select * from Wine",conn)
        df = df.groupby('quality')['volatileacidity'].describe()
        model = PandasModel(df)
        self.ui.tableView_wine_detail_5.setModel(model)

    def newexpresult(self):
        conn = sqlite3.connect('veri_tabanı.db')
        df = pd.read_sql_query("select * from Wine",conn)
        y=df["quality"]
        X=df.drop(["quality"],axis=1)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
        model = RandomForestClassifier(max_features = 11,min_samples_split =5,n_estimators=10).fit(X_train,y_train)
        fixa = float(self.ui.lineEdit_fixed.text())
        volat = float(self.ui.lineEdit_volatile.text())
        cit = float(self.ui.lineEdit_acid.text())
        sugar = float(self.ui.lineEdit_sugar.text())
        chlo = float(self.ui.lineEdit_chlo.text())
        free = float(self.ui.lineEdit_freesu.text())
        total = float(self.ui.lineEdit_total.text())
        denst = float(self.ui.lineEdit_density.text())
        ph = float(self.ui.lineEdit_ph.text())
        suph = float(self.ui.lineEdit_sulpha.text())
        alch = float(self.ui.lineEdit_alcohol.text())
        new_exp = [[fixa],[volat],[cit],[sugar],[chlo],[free],[total],[denst],[ph],[suph],[alch]]
        new_exp = pd.DataFrame(new_exp).T
        pre = model.predict(new_exp)
        pre = pre.item(0)
        st_pre = str(pre)
        self.ui.lineEdit_segment.setText(st_pre)
    
    def importance(self):
        conn = sqlite3.connect('veri_tabanı.db')
        df = pd.read_sql_query("select * from Wine",conn)
        y=df["quality"]
        X=df.drop(["quality"],axis=1)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=42)
        model = RandomForestClassifier(max_features = 11,min_samples_split =5,n_estimators=10).fit(X_train,y_train)
        df = pd.DataFrame({'Features':X_train.columns})
        df['%_Importance'] =(model.feature_importances_*100).astype(int)
        labels = df['Features'].tolist()
        sizes = df['%_Importance'].tolist()
        colors = ['#ff9999','#66b3ff','#ff99e6','#ffcc99']
        self.ui.MplWidget_7.canvas.axes.clear()
        self.ui.MplWidget_7.canvas.axes.pie(sizes,labels=labels,autopct='%1.0f%%',radius=.9,startangle = 90,colors=colors,explode=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),shadow=True)
        self.ui.MplWidget_7.canvas.axes.axis('equal')
        #self.ui.MplWidget_7.canvas.axes.legend(labels, loc='upper right')
        self.ui.MplWidget_7.canvas.axes.set_title('Feature Importance Pie Chart',weight='bold')
        self.ui.MplWidget_7.canvas.draw()
        
        
    def succesrate(self):
        conn = sqlite3.connect('veri_tabanı.db')
        df = pd.read_sql_query("select * from Wine",conn)
        y=df["quality"]
        X=df.drop(["quality"],axis=1)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
        cvs = self.ui.lineEdit_cvclass.text()
        if cvs == "":
            QMessageBox.information(self, "Info", 'Please insert Cross Validation to calculate success rates!')
        else:
            ##cart
            cart_model = DecisionTreeClassifier().fit(X_train,y_train)
            cvc = float('%.3f' % cross_val_score(cart_model,X_test,y_test,cv=int(cvs)).mean())*100
            st_cvc = str(round(cvc,2))
            self.ui.lineEdit_desctree.setText(st_cvc)
            ##lojistic
            loj_model = LogisticRegression(solver="liblinear").fit(X_train,y_train)
            cvl = float('%.3f' % cross_val_score(loj_model,X_test,y_test,cv=int(cvs)).mean())*100
            st_cvl = str(round(cvl,2))
            self.ui.lineEdit_lojreg.setText(st_cvl)
            ##Kneighbours
            knn_model = KNeighborsClassifier(n_neighbors = 1).fit(X_train, y_train)
            cvk = float('%.3f' % cross_val_score(knn_model,X_test,y_test,cv=int(cvs)).mean())*100
            st_cvk = str(round(cvk,2))
            self.ui.lineEdit_kneigh.setText(st_cvk)
            ##random forest
            rfor_model = RandomForestClassifier(max_features = 11,min_samples_split =5,n_estimators=10).fit(X_train,y_train)
            rfor_pred = rfor_model.predict(X_test)
            rfor_acc = float('%.3f' % accuracy_score(y_test,rfor_pred))*100
            st_rfor_acc = str(round(rfor_acc,2))
            self.ui.lineEdit_random.setText(st_rfor_acc)
            ##LDA
            lda_model = LDA().fit(X_train,y_train)
            cvl = float('%.3f' % cross_val_score(lda_model,X_test,y_test,cv=int(cvs)).mean())*100
            st_cvl = str(round(cvl,2))
            self.ui.lineEdit_mlp.setText(st_cvl)
            
    def clearwine(self):
        self.ui.lineEdit_fixed.setText('')
        self.ui.lineEdit_volatile.setText('')
        self.ui.lineEdit_acid.setText('')
        self.ui.lineEdit_sugar.setText('')
        self.ui.lineEdit_chlo.setText('')
        self.ui.lineEdit_freesu.setText('')
        self.ui.lineEdit_total.setText('')
        self.ui.lineEdit_density.setText('')
        self.ui.lineEdit_ph.setText('')
        self.ui.lineEdit_sulpha.setText('')
        self.ui.lineEdit_alcohol.setText('')

##customer segment##
        
    def deleteretail(self):
        selected = str(self.ui.comboBox.currentText())
        invoice = str(self.ui.lineEdit_invoice.text())
        customer = str(self.ui.lineEdit_customer.text())
        stock = str(self.ui.lineEdit_stock.text())
        stockdescr = str(self.ui.lineEdit_stock_desc.text())
        date = str(self.ui.lineEdit_invodate.text())
        dateb = str(self.ui.lineEdit_until.text())
        datea = str(self.ui.lineEdit_until_2.text())
        conn = sqlite3.connect('veri_tabanı.db')
        if (invoice=="" and customer=="" and stock=="" and stockdescr=="" and date=="" and dateb=="" and datea==""):
            QMessageBox.critical(self, "Opps!", 'You can delete a record after filtering!')
        if invoice !="":
            df = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}' and InvoiceNo like '{invoice}%'", conn)
            leng = "Do you want to delete " + str(len(df)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from OnlineRetails where Country = '{selected}' and InvoiceNo like '{invoice}%'")
                conn.commit()
                suc = "Country: "+selected+"Invoice: "+invoice+" "+str(len(df))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if customer != "":
            df = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}' and CustomerID like '{customer}%'", conn)
            leng = "Do you want to delete " + str(len(df)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from OnlineRetails where Country = '{selected}' and CustomerID like '{customer}%'")
                conn.commit()
                suc = "Country: "+selected+" Customer: "+customer+" "+str(len(df))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if stock != "":
            df = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}' and StockCode like '{stock}%'", conn)
            leng = "Do you want to delete " + str(len(df)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from OnlineRetails where Country = '{selected}' and StockCode like '{stock}%'")
                conn.commit()
                suc = "Country: "+selected+" Stok: "+stock+" "+str(len(df))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if invoice != "" and customer != "":
            df = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}' and InvoiceNo like '{invoice}%' and CustomerID like '{customer}%'", conn)
            leng = "Do you want to delete " + str(len(df)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from OnlineRetails where Country = '{selected}' and InvoiceNo like '{invoice}%' and CustomerID like '{customer}%'")
                conn.commit()
                suc = "Country: "+selected+" Customer: "+customer+", Invoice: "+invoice+" "+str(len(df))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if invoice != "" and customer != "" and stock != "":
            df = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}' and InvoiceNo like '{invoice}%' and CustomerID like '{customer}%'", conn)
            leng = "Do you want to delete " + str(len(df)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from OnlineRetails where Country = '{selected}' and InvoiceNo like '{invoice}%' and CustomerID like '{customer}%' and StockCode like '{stock}%'")
                conn.commit()
                suc = "Country: "+selected+" Customer: "+customer+", Stok: "+stock+" "+str(len(df))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if invoice != "" and customer != "" and stock != "":
            df = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}' and InvoiceNo like '{invoice}%' and CustomerID like '{customer}%'", conn)
            leng = "Do you want to delete " + str(len(df)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from OnlineRetails where Country = '{selected}' and InvoiceNo like '{invoice}%' and CustomerID like '{customer}%' and StockCode like '{stock}%'")
                conn.commit()
                suc = "Country: "+selected+" Customer: "+customer+", Stok: "+stock+" "+str(len(df))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if customer != "" and stock != "":
            df = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}' and CustomerID like '{customer}%' and StockCode like '{stock}%'", conn)
            leng = "Do you want to delete " + str(len(df)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from OnlineRetails where Country = '{selected}' and CustomerID like '{customer}%' and StockCode like '{stock}%'")
                conn.commit()
                suc = "Country: "+selected+" Customer: "+customer+", Invoice: "+invoice+" "+"Stok: "+stock+" "+str(len(df))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if  invoice != "" and stock != "":
            df = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}' and InvoiceNo like '{invoice}%' and StockCode like '{stock}%'", conn)
            leng = "Do you want to delete " + str(len(df)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from OnlineRetails where Country = '{selected}' and InvoiceNo like '{invoice}%' and StockCode like '{stock}%'")
                conn.commit()
                suc = "Country: "+selected+", Invoice: "+invoice+" "+"Stok: "+stock+" "+str(len(df))+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if  stockdescr != "":
            df = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}' and Description like '{stockdescr}%'", conn)
            leng = "Do you want to delete " + str(len(df)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from OnlineRetails where Country = '{selected}' and Description like '{stockdescr}%'")
                conn.commit()
                suc = "Country: "+selected+" Stock Description: "+stockdescr+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if  date!= "": 
            df = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}' and InvoiceDate like '{date}%'", conn)
            leng = "Do you want to delete " + str(len(df)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from OnlineRetails where Country = '{selected}' and InvoiceDate like '{date}%'")
                conn.commit()
                suc = "Date: "+date+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
        if  dateb!= "" and datea!= "":
            df = pd.read_sql_query(f"select * from OnlineRetails where (substr(InvoiceDate, 7, 4) || '-' || substr(InvoiceDate, 4, 2) || '-' || substr(InvoiceDate, 1, 2)) between '{dateb}' and '{datea}' and Country = '{selected}'", conn)
            leng = "Do you want to delete " + str(len(df)) + " record?"
            buttonReply = QMessageBox.question(self, 'Request?', leng, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                conn.execute(f"delete from OnlineRetails where (substr(InvoiceDate, 7, 4) || '-' || substr(InvoiceDate, 4, 2) || '-' || substr(InvoiceDate, 1, 2)) between '{dateb}' and '{datea}' and Country = '{selected}'")
                conn.commit()
                suc = "Begin Date: "+dateb+" End Date: "+datea+" row deleted!"
                QMessageBox.information(self, "Deleted Successfully!",suc)  
            else:
                QMessageBox.information(self, "Aborted!", 'Delete request aborted!')
    
    def displayretail(self):
        selected = self.ui.comboBox.currentText()
        invoice = self.ui.lineEdit_invoice.text()
        customer = self.ui.lineEdit_customer.text()
        stock = self.ui.lineEdit_stock.text()
        stockdescr = self.ui.lineEdit_stock_desc.text()
        date = self.ui.lineEdit_invodate.text()
        dateb = self.ui.lineEdit_until.text()
        datea = self.ui.lineEdit_until_2.text()
        conn = sqlite3.connect('veri_tabanı.db')
        tx_data = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}'", conn)
        modeldis = PandasModel(tx_data)
        self.ui.tableView_retail_data.setModel(modeldis)
        self.ui.label_count.setText('Count:'+ str(len(tx_data)))
        if invoice != "":
            dfin = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}' and InvoiceNo like '{invoice}%'", conn)
            modeli = PandasModel(dfin)
            self.ui.tableView_retail_data.setModel(modeli)
            self.ui.label_count.setText('Count:'+ str(len(dfin)))
        if customer != "":
            dfcus = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}' and CustomerID like '{customer}%'", conn)
            modelc = PandasModel(dfcus)
            self.ui.tableView_retail_data.setModel(modelc)
            self.ui.label_count.setText('Count:'+ str(len(dfcus)))
        if stock != "":
            dfsto = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}' and StockCode like '{stock}%'", conn)
            modelc = PandasModel(dfsto)
            self.ui.tableView_retail_data.setModel(modelc)
            self.ui.label_count.setText('Count:'+ str(len(dfsto)))
        if invoice != "" and customer != "":
            dftwo = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}' and InvoiceNo like '{invoice}%' and CustomerID like '{customer}%'", conn)
            modelt = PandasModel(dftwo)
            self.ui.tableView_retail_data.setModel(modelt)
            self.ui.label_count.setText('Count:'+ str(len(dftwo)))
        if invoice != "" and customer != "" and stock != "":
            dftwo = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}' and InvoiceNo like '{invoice}%' and CustomerID like '{customer}%' and StockCode like '{stock}%'", conn)
            modelt = PandasModel(dftwo)
            self.ui.tableView_retail_data.setModel(modelt)
            self.ui.label_count.setText('Count:'+ str(len(dftwo)))
        if  customer != "" and stock != "":
            dftwo = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}' and CustomerID like '{customer}%' and StockCode like '{stock}%'", conn)
            modelt = PandasModel(dftwo)
            self.ui.tableView_retail_data.setModel(modelt)
            self.ui.label_count.setText('Count:'+ str(len(dftwo)))
        if  invoice != "" and stock != "":
            dftwo = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}' and InvoiceNo like '{invoice}%' and StockCode like '{stock}%'", conn)
            modelt = PandasModel(dftwo)
            self.ui.tableView_retail_data.setModel(modelt)
            self.ui.label_count.setText('Count:'+ str(len(dftwo)))
        if  stockdescr != "":
            dftwo = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}' and Description like '{stockdescr}%'", conn)
            modelt = PandasModel(dftwo)
            self.ui.tableView_retail_data.setModel(modelt)
            self.ui.label_count.setText('Count:'+ str(len(dftwo)))
        if  date!= "":
            dftdatef = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}' and InvoiceDate like '{date}%'", conn)
            modelt = PandasModel(dftdatef)
            self.ui.tableView_retail_data.setModel(modelt)
            self.ui.label_count.setText('Count:'+ str(len(dftdatef)))
        if  dateb!= "" and datea!= "":
            dftdate = pd.read_sql_query(f"select * from OnlineRetails where (substr(InvoiceDate, 7, 4) || '-' || substr(InvoiceDate, 4, 2) || '-' || substr(InvoiceDate, 1, 2)) between '{dateb}' and '{datea}' and Country = '{selected}'", conn)
            modelt = PandasModel(dftdate)
            self.ui.tableView_retail_data.setModel(modelt)
            self.ui.label_count.setText('Count:'+ str(len(dftdate)))
    
    def exportdisplayed(self):
        selected = self.ui.comboBox.currentText()
        invoice = self.ui.lineEdit_invoice.text()
        customer = self.ui.lineEdit_customer.text()
        stock = self.ui.lineEdit_stock.text()
        stockdescr = self.ui.lineEdit_stock_desc.text()
        date = self.ui.lineEdit_invodate.text()
        dateb = self.ui.lineEdit_until.text()
        datea = self.ui.lineEdit_until_2.text()
        conn = sqlite3.connect('veri_tabanı.db')
        tx_data = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}'", conn) 
        if invoice == "" and customer =="" and stock =="" and stockdescr =="" and date =="" and dateb =="" and datea =="":
            try:
                bookname = 'Filtered_'+selected+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'All_Records'
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                tx_data.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Online Retail Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        if invoice != "" and customer =="" and stock =="" and stockdescr =="" and date =="" and dateb =="" and datea =="":
            dfin = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}' and InvoiceNo like '{invoice}%'", conn)
            try:
                bookname = 'Filtered_'+selected+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'Invoice_with_'+invoice
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                dfin.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Online Retail Invoice Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        if customer != "" and invoice == ""  and stock =="" and stockdescr =="" and date =="" and dateb =="" and datea =="":
            dfcus = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}' and CustomerID like '{customer}%'", conn)
            try:
                bookname = 'Filtered_'+selected+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'Customer_with_'+customer
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                dfcus.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Online Retail Customer Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        if stock != "" and invoice == "" and customer =="" and stockdescr =="" and date =="" and dateb =="" and datea =="":
            dfsto = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}' and StockCode like '{stock}%'", conn)
            try:
                bookname = 'Filtered_'+selected+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'Stock_with_'+stock
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                dfsto.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Online Retail Stock Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        if invoice != "" and customer != "" and stock =="" and stockdescr =="" and date =="" and dateb =="" and datea =="":
            dftwo = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}' and InvoiceNo like '{invoice}%' and CustomerID like '{customer}%'", conn)
            try:
                bookname = 'Filtered_'+selected+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = invoice+'_'+customer
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                dftwo.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Online Retail invoice and customer Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        if invoice != "" and customer != "" and stock != "" and stockdescr =="" and date =="" and dateb =="" and datea =="":
            dftwo = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}' and InvoiceNo like '{invoice}%' and CustomerID like '{customer}%' and StockCode like '{stock}%'", conn)
            try:
                bookname = 'Filtered_'+selected+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = invoice+'_'+stock
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                dftwo.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Online Retail invoice,customer and stock Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        if customer != "" and stock != "" and invoice == "" and stockdescr =="" and date =="" and dateb =="" and datea =="":
            dftwo = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}' and CustomerID like '{customer}%' and StockCode like '{stock}%'", conn)
            try:
                bookname = 'Filtered_'+selected+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = customer+'_'+stock
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                dftwo.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Online Retail customer and stock Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        if  invoice != "" and stock != "" and customer =="" and stockdescr =="" and date =="" and dateb =="" and datea =="":
            dftwo = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}' and InvoiceNo like '{invoice}%' and StockCode like '{stock}%'", conn)
            try:
                bookname = 'Filtered_'+selected+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = invoice+'_'+stock
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                dftwo.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Online Retail invoice and stock Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        if  stockdescr != "" and invoice == "" and customer =="" and stock =="" and date =="" and dateb =="" and datea =="":
            dftwo = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}' and Description like '{stockdescr}%'", conn)
            try:
                bookname = 'Filtered_'+selected+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'Desc_with_'+stockdescr
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                dftwo.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Online Retail stock description Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
        if  date != "" and invoice == "" and customer =="" and stock =="" and stockdescr =="" and dateb =="" and datea =="":
            dftdatef = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}' and InvoiceDate like '{date}%'", conn)
            try:
                bookname = 'Filtered_'+selected+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'In_the_'+date
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                dftdatef.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Online Retail Data has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')  
        if  dateb!= "" and datea!= "" and invoice == "" and customer =="" and stock =="" and stockdescr =="" and date =="":
            dftdate = pd.read_sql_query(f"select * from OnlineRetails where (substr(InvoiceDate, 7, 4) || '-' || substr(InvoiceDate, 4, 2) || '-' || substr(InvoiceDate, 1, 2)) between '{dateb}' and '{datea}' and Country = '{selected}'", conn)
            try:
                bookname = 'Filtered_'+selected+ time.strftime("_%m_%d_%Y") +'.xlsx'
                sheetname = 'between_'+dateb+'and_'+datea
                datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
                dftdate.to_excel(datatoexcel,sheet_name=sheetname)
                datatoexcel.save()
                QMessageBox.information(self, "Success!", 'Online Retail Data between dates has been exported!')
            except sqlite3.Error as error:
                QMessageBox.critical(self, "Error!", 'Could not export!')
                
    def clearsearchretail(self):
        self.ui.comboBox.setCurrentText('')
        self.ui.lineEdit_invoice.setText('')
        self.ui.lineEdit_customer.setText('')
        self.ui.lineEdit_stock.setText('')
        self.ui.lineEdit_stock_desc.setText('')
        self.ui.lineEdit_invodate.setText('')
        self.ui.lineEdit_until.setText('')
        self.ui.lineEdit_until_2.setText('')
        
    
    def overall(self):
        selected = self.ui.comboBox_2.currentText()
        conn = sqlite3.connect('veri_tabanı.db')
        tx_data = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}'", conn)
        tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'])
        tx_select = tx_data.query(f"Country =='{selected}'").reset_index(drop=True)
        tx_user = pd.DataFrame(tx_select['CustomerID'].unique())
        tx_user.columns = ['CustomerID']
        tx_max_purchase = tx_select.groupby('CustomerID').InvoiceDate.max().reset_index()
        tx_max_purchase.columns = ['CustomerID','MaxPurchaseDate']
        tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days
        tx_user = pd.merge(tx_user, tx_max_purchase[['CustomerID','Recency']], on='CustomerID')
        #recency
        kmeans = KMeans(n_clusters=4)
        kmeans.fit(tx_user[['Recency']])
        tx_user['RecencyCluster'] = kmeans.predict(tx_user[['Recency']])
        
        def order_cluster(cluster_field_name, target_field_name,df,ascending):
            df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
            df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
            df_new['index'] = df_new.index
            df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
            df_final = df_final.drop([cluster_field_name],axis=1)
            df_final = df_final.rename(columns={"index":cluster_field_name})
            return df_final
        
        tx_user = order_cluster('RecencyCluster', 'Recency',tx_user,False)
        #frequency
        tx_frequency = tx_select.groupby('CustomerID').InvoiceNo.count().reset_index()
        tx_frequency.columns = ['CustomerID','Frequency']
        tx_user = pd.merge(tx_user, tx_frequency, on='CustomerID')
        
        kmeans = KMeans(n_clusters=4)
        kmeans.fit(tx_user[['Frequency']])
        tx_user['FrequencyCluster'] = kmeans.predict(tx_user[['Frequency']])
        tx_user = order_cluster('FrequencyCluster', 'Frequency',tx_user,True)
        #revenue
        tx_select['Revenue'] = tx_select['UnitPrice'].astype(float) * tx_select['Quantity'].astype(float)
        tx_revenue = tx_select.groupby('CustomerID').Revenue.sum().reset_index()
        tx_user = pd.merge(tx_user, tx_revenue, on='CustomerID')
        
        kmeans = KMeans(n_clusters=4)
        kmeans.fit(tx_user[['Revenue']])
        tx_user['RevenueCluster'] = kmeans.predict(tx_user[['Revenue']])
        tx_user = order_cluster('RevenueCluster', 'Revenue',tx_user,True)
        tx_user['OverallScore'] = tx_user['RecencyCluster'] + tx_user['FrequencyCluster'] + tx_user['RevenueCluster']
        tx_user = tx_user.drop(['Recency','RecencyCluster','Frequency','FrequencyCluster','Revenue','RevenueCluster'],axis=1)
        tx_user['Segment'] = 'Low-Value'
        tx_user.loc[tx_user['OverallScore']>2,'Segment'] = 'Mid-Value' 
        tx_user.loc[tx_user['OverallScore']>4,'Segment'] = 'High-Value' 
        modelt = PandasModel(tx_user)
        self.ui.tableView_onlinescore.setModel(modelt)
        tx_low = tx_user.query("Segment == 'Low-Value'")
        tx_mid = tx_user.query("Segment == 'Mid-Value'")
        tx_high = tx_user.query("Segment == 'High-Value'")
        low_df = tx_low.pivot_table(index='Segment',aggfunc='size').reset_index()
        mid_df = tx_mid.pivot_table(index='Segment',aggfunc='size').reset_index()
        high_df = tx_high.pivot_table(index='Segment',aggfunc='size').reset_index()
        low_df.columns = ['Segment','Counts']
        mid_df.columns = ['Segment','Counts']
        high_df.columns = ['Segment','Counts']
        self.ui.MplWidget_score_3.canvas.axes.clear()
        self.ui.MplWidget_score_3.canvas.axes.bar('Segment','Counts',data=low_df,color='orange')
        self.ui.MplWidget_score_3.canvas.axes.bar('Segment','Counts',data=mid_df)
        self.ui.MplWidget_score_3.canvas.axes.bar('Segment','Counts',data=high_df,color='green')
        self.ui.MplWidget_score_3.canvas.axes.set_title('Segments by Customers')
        self.ui.MplWidget_score_3.canvas.axes.set_xlabel('Segments')
        self.ui.MplWidget_score_3.canvas.axes.set_ylabel('Counts')
        self.ui.MplWidget_score_3.canvas.draw()
    
    def exportoverall(self):
        selected = self.ui.comboBox_2.currentText()
        conn = sqlite3.connect('veri_tabanı.db')
        tx_data = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}'", conn)
        tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'])
        tx_select = tx_data.query(f"Country =='{selected}'").reset_index(drop=True)
        tx_user = pd.DataFrame(tx_select['CustomerID'].unique())
        tx_user.columns = ['CustomerID']
        tx_max_purchase = tx_select.groupby('CustomerID').InvoiceDate.max().reset_index()
        tx_max_purchase.columns = ['CustomerID','MaxPurchaseDate']
        tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days
        tx_user = pd.merge(tx_user, tx_max_purchase[['CustomerID','Recency']], on='CustomerID')
        #recency
        kmeans = KMeans(n_clusters=4)
        kmeans.fit(tx_user[['Recency']])
        tx_user['RecencyCluster'] = kmeans.predict(tx_user[['Recency']])
        
        def order_cluster(cluster_field_name, target_field_name,df,ascending):
            df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
            df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
            df_new['index'] = df_new.index
            df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
            df_final = df_final.drop([cluster_field_name],axis=1)
            df_final = df_final.rename(columns={"index":cluster_field_name})
            return df_final
        
        tx_user = order_cluster('RecencyCluster', 'Recency',tx_user,False)
        #frequency
        tx_frequency = tx_select.groupby('CustomerID').InvoiceNo.count().reset_index()
        tx_frequency.columns = ['CustomerID','Frequency']
        tx_user = pd.merge(tx_user, tx_frequency, on='CustomerID')
        
        kmeans = KMeans(n_clusters=4)
        kmeans.fit(tx_user[['Frequency']])
        tx_user['FrequencyCluster'] = kmeans.predict(tx_user[['Frequency']])
        tx_user = order_cluster('FrequencyCluster', 'Frequency',tx_user,True)
        #revenue
        tx_select['Revenue'] = tx_select['UnitPrice'].astype(float) * tx_select['Quantity'].astype(float)
        tx_revenue = tx_select.groupby('CustomerID').Revenue.sum().reset_index()
        tx_user = pd.merge(tx_user, tx_revenue, on='CustomerID')
        
        kmeans = KMeans(n_clusters=4)
        kmeans.fit(tx_user[['Revenue']])
        tx_user['RevenueCluster'] = kmeans.predict(tx_user[['Revenue']])
        tx_user = order_cluster('RevenueCluster', 'Revenue',tx_user,True)
        tx_user['OverallScore'] = tx_user['RecencyCluster'] + tx_user['FrequencyCluster'] + tx_user['RevenueCluster']
        tx_user = tx_user.drop(['Recency','RecencyCluster','Frequency','FrequencyCluster','Revenue','RevenueCluster'],axis=1)
        tx_user['Segment'] = 'Low-Value'
        tx_user.loc[tx_user['OverallScore']>2,'Segment'] = 'Mid-Value' 
        tx_user.loc[tx_user['OverallScore']>4,'Segment'] = 'High-Value' 
        bookname = 'Overall_Segments'+ time.strftime("_%m_%d_%Y") +'.xlsx'
        sheetname = 'Overall'
        datatoexcel = pd.ExcelWriter(bookname,engine='xlsxwriter')
        tx_user.to_excel(datatoexcel,sheet_name=sheetname)
        datatoexcel.save()
        QMessageBox.information(self, "Success!", 'Segment based overall score data exported!')
        
    def scatterplotseg(self):
        selected = self.ui.comboBox_2.currentText()
        conn = sqlite3.connect('veri_tabanı.db')
        tx_data = pd.read_sql_query(f"select * from OnlineRetails where Country = '{selected}'", conn)
        tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'])
        tx_select = tx_data.query(f"Country =='{selected}'").reset_index(drop=True)
        tx_user = pd.DataFrame(tx_select['CustomerID'].unique())
        tx_user.columns = ['CustomerID']
        tx_max_purchase = tx_select.groupby('CustomerID').InvoiceDate.max().reset_index()
        tx_max_purchase.columns = ['CustomerID','MaxPurchaseDate']
        tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days
        tx_user = pd.merge(tx_user, tx_max_purchase[['CustomerID','Recency']], on='CustomerID')
        #recency
        kmeans = KMeans(n_clusters=4)
        kmeans.fit(tx_user[['Recency']])
        tx_user['RecencyCluster'] = kmeans.predict(tx_user[['Recency']])
        
        def order_cluster(cluster_field_name, target_field_name,df,ascending):
            df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
            df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
            df_new['index'] = df_new.index
            df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
            df_final = df_final.drop([cluster_field_name],axis=1)
            df_final = df_final.rename(columns={"index":cluster_field_name})
            return df_final
        
        tx_user = order_cluster('RecencyCluster', 'Recency',tx_user,False)
        #frequency
        tx_frequency = tx_select.groupby('CustomerID').InvoiceNo.count().reset_index()
        tx_frequency.columns = ['CustomerID','Frequency']
        tx_user = pd.merge(tx_user, tx_frequency, on='CustomerID')
        
        kmeans = KMeans(n_clusters=4)
        kmeans.fit(tx_user[['Frequency']])
        tx_user['FrequencyCluster'] = kmeans.predict(tx_user[['Frequency']])
        tx_user = order_cluster('FrequencyCluster', 'Frequency',tx_user,True)
        #revenue
        tx_select['Revenue'] = tx_select['UnitPrice'].astype(float) * tx_select['Quantity'].astype(float)
        tx_revenue = tx_select.groupby('CustomerID').Revenue.sum().reset_index()
        tx_user = pd.merge(tx_user, tx_revenue, on='CustomerID')
        
        kmeans = KMeans(n_clusters=4)
        kmeans.fit(tx_user[['Revenue']])
        tx_user['RevenueCluster'] = kmeans.predict(tx_user[['Revenue']])
        tx_user = order_cluster('RevenueCluster', 'Revenue',tx_user,True)
        tx_user['OverallScore'] = tx_user['RecencyCluster'] + tx_user['FrequencyCluster'] + tx_user['RevenueCluster']
        #groupover= tx_user.groupby('OverallScore')['Recency','Frequency','Revenue'].mean()
        #tx_user = tx_user.drop(['Recency','RecencyCluster','Frequency','FrequencyCluster','Revenue','RevenueCluster'],axis=1)
        tx_user['Segment'] = 'Low-Value'
        tx_user.loc[tx_user['OverallScore']>2,'Segment'] = 'Mid-Value' 
        tx_user.loc[tx_user['OverallScore']>4,'Segment'] = 'High-Value'
        tx_low = tx_user.query("Segment == 'Low-Value'")
        tx_mid = tx_user.query("Segment == 'Mid-Value'")
        tx_high = tx_user.query("Segment == 'High-Value'")
        self.ui.MplWidget_score.canvas.axes.clear()
        self.ui.MplWidget_score.canvas.axes.scatter(x='Revenue',y='CustomerID',s=75,data=tx_mid,alpha=0.5)
        self.ui.MplWidget_score.canvas.axes.scatter(x='Revenue',y='CustomerID',color= 'orange',s=50,data=tx_low,alpha=0.5)
        self.ui.MplWidget_score.canvas.axes.scatter(x='Revenue',y='CustomerID',color= 'green',s=100,data=tx_high,alpha=0.7)
        self.ui.MplWidget_score.canvas.axes.set_yticklabels([])
        self.ui.MplWidget_score.canvas.axes.legend(('MIDDLE','LOW','HIGH'),loc='upper right')
        self.ui.MplWidget_score.canvas.axes.set_title('Segment Scatter',fontsize=10,weight='bold')
        self.ui.MplWidget_score.canvas.axes.set_xlabel('Revenue')
        self.ui.MplWidget_score.canvas.axes.set_ylabel('Customers')
        self.ui.MplWidget_score.canvas.draw()
        self.ui.MplWidget_score_2.canvas.axes.clear()
        self.ui.MplWidget_score_2.canvas.axes.scatter(x='Frequency',y='Revenue',s=75,data=tx_mid,alpha=0.5)
        self.ui.MplWidget_score_2.canvas.axes.scatter(x='Frequency',y='Revenue',color= 'orange',s=50,data=tx_low,alpha=0.5)
        self.ui.MplWidget_score_2.canvas.axes.scatter(x='Frequency',y='Revenue',color= 'green',s=100,data=tx_high,alpha=0.7)
        self.ui.MplWidget_score_2.canvas.axes.legend(('MIDDLE','LOW','HIGH'),loc='upper right')
        self.ui.MplWidget_score_2.canvas.axes.set_title('Frequency-Revenue Scatter',fontsize=10,weight='bold')
        self.ui.MplWidget_score_2.canvas.axes.set_xlabel('Frequency')
        self.ui.MplWidget_score_2.canvas.axes.set_ylabel('Revenue')
        self.ui.MplWidget_score_2.canvas.draw()
        self.ui.MplWidget_score_4.canvas.axes.clear()
        self.ui.MplWidget_score_4.canvas.axes.scatter(x='Recency',y='Revenue',s=75,data=tx_mid,alpha=0.5)
        self.ui.MplWidget_score_4.canvas.axes.scatter(x='Recency',y='Revenue',color= 'orange',s=50,data=tx_low,alpha=0.5)
        self.ui.MplWidget_score_4.canvas.axes.scatter(x='Recency',y='Revenue',color= 'green',s=100,data=tx_high,alpha=0.7)
        self.ui.MplWidget_score_4.canvas.axes.legend(('MIDDLE','LOW','HIGH'),loc='upper right')
        self.ui.MplWidget_score_4.canvas.axes.set_title('Recency-Revenue Scatter',fontsize=10,weight='bold')
        self.ui.MplWidget_score_4.canvas.axes.set_xlabel('Recency')
        self.ui.MplWidget_score_4.canvas.axes.set_ylabel('Revenue')
        self.ui.MplWidget_score_4.canvas.draw()
        
        
    def arffregreport(self):
        rcv = int(self.ui.lineEdit_reg_cv.text())
        fileNameTup = QFileDialog.getOpenFileName(self, 'OpenFile',"", "Arff (*.arff)")
        fileNamea = fileNameTup[0]
        if fileNamea != "":
            data = arff.loadarff(open(fileNamea))
            df = pd.DataFrame(data[0])
            df = df.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)#displayed fixed categorical values
            modeldis = PandasModel(df)
            self.ui.tableView_reg_arff.setModel(modeldis)
            cols = df.columns.tolist()
            lastone = str(cols[-1])
            lastonetype = df.dtypes[lastone]
            pct = df.pivot_table(index=[lastone],aggfunc='size')
            df_co = pct.reset_index()
            df_co.columns = ['class','counts']
            #ignore possib.<2 frequency for classf.
            if (lastonetype == np.object and len(df_co)<15) or (lastonetype == np.float64 and len(df_co)<15): #tested with clas.datasets optimum threshold '10' for the counts of classes
                QMessageBox.information(self, "Info", 'Your dataset is Classification!')
                df[lastone] = df[lastone].astype('category')
                df['encoded_labels'] = df[lastone].cat.codes #last column encoded 
                y = df.iloc[:,-1:]
                X= df.drop(df.iloc[:,-2:],axis=1)
                cat_fea_mask = X.dtypes==np.object
                ##for categorical attributes
                categorical_cols = X.columns[cat_fea_mask].tolist()
                le = LabelEncoder()
                X[categorical_cols] = X[categorical_cols].apply(lambda col: le.fit_transform(col))
                X_ = X.drop(X.columns[cat_fea_mask],axis=1)
                X_ = X_.apply(lambda x: x.fillna(int(x.mean())),axis= 0)
                X = pd.concat([X[categorical_cols],X_],axis=1)
                X = X.reindex(sorted(X.columns), axis=1)
                ##split
                X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
                ##decisiontree
                cart_model = DecisionTreeClassifier().fit(X_train,y_train)
                cart = float('%.3f' % cross_val_score(cart_model,X_test,y_test,cv=rcv).mean())*100
                cart_str = str(round(cart,2))
                self.ui.lineEdit_desctree_5.setText(cart_str)
                ##lojistic
                loj_model = LogisticRegression(solver="liblinear").fit(X_train,y_train)
                loj = float('%.3f' % cross_val_score(loj_model,X_test,y_test,cv=rcv).mean())*100
                loj_str = str(round(loj,2))
                self.ui.lineEdit_lojreg_5.setText(loj_str)
                ##Kneighbours
                knn_model = KNeighborsClassifier(n_neighbors = 2).fit(X_train, y_train)
                knn = float('%.3f' % cross_val_score(knn_model,X_test,y_test,cv=rcv).mean())*100
                knn_str = str(round(knn,2))
                self.ui.lineEdit_kneigh_5.setText(knn_str)
                ##randomforest
                rfor_model = RandomForestClassifier().fit(X_train,y_train)
                rfor_pred = rfor_model.predict(X_test)
                rfor_acc = float('%.3f' % accuracy_score(y_test,rfor_pred))*100
                rfor_acc_str = str(round(rfor_acc,2))
                self.ui.lineEdit_random_5.setText(rfor_acc_str)
                ##LDA
                lda_model = LDA().fit(X_train,y_train)
                lda = float('%.3f' % cross_val_score(lda_model,X_test,y_test,cv=rcv).mean())*100
                lda_str = str(round(lda,2))
                self.ui.lineEdit_mlp_5.setText(lda_str)
                #importance
                model = RandomForestClassifier().fit(X_train,y_train)
                dfi = pd.DataFrame({'Features':X_train.columns})
                dfi['Importance'] =(model.feature_importances_*100).round(4)
                dfi['Features'] = dfi['Features'].astype('category')
                dfi['Encoded_fea'] = dfi['Features'].cat.codes
                model_s = PandasModel(dfi)
                self.ui.tableView_class_imp.setModel(model_s)
                #max importance
                mx = dfi[dfi.Importance == dfi.Importance.max()]
                li = mx['Features'].tolist()
                fea = str(li[0])
                self.ui.lineEdit_max_imp.setText(fea)
                #bar
                self.ui.MplWidget_scat_clas.canvas.axes.clear()
                self.ui.MplWidget_scat_clas.canvas.axes.bar('Encoded_fea','Importance',data = dfi,color ='#4dc0d4')
                self.ui.MplWidget_scat_clas.canvas.axes.legend(('Importance','Features'),loc='upper right')
                self.ui.MplWidget_scat_clas.canvas.axes.set_title('Classification BarPlot',fontsize=10,weight='bold')
                self.ui.MplWidget_scat_clas.canvas.axes.set_xlabel('Encoded_fea')
                self.ui.MplWidget_scat_clas.canvas.axes.set_ylabel('Importance')
                self.ui.MplWidget_scat_clas.canvas.draw()
            if (lastonetype == np.float64 and lastonetype != np.object and len(df_co)>15):
                QMessageBox.information(self, "Info", 'Your dataset is Regression!')
                y = df.iloc[:,-1:]
                X= df.drop(df.iloc[:,-1:],axis=1)
                X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
                #linearreg
                lin_model = LinearRegression().fit(X_train,y_train)
                lin_mse = str(float('%.3f' % np.mean(-cross_val_score(lin_model,X_train,y_train,cv= rcv, scoring= "neg_mean_squared_error"))))
                self.ui.lineEdit_mse_linreg.setText(lin_mse)
                lin_rmse = str(float('%.3f' % np.sqrt(np.mean(-cross_val_score(lin_model,X_train,y_train, cv= rcv, scoring= "neg_mean_squared_error")))))
                self.ui.lineEdit_rmse_linreg.setText(lin_rmse)
                #LassoReg
                las_model = Lasso().fit(X_train,y_train)
                las_mse = str(float('%.3f' % np.mean(-cross_val_score(las_model,X_train,y_train,cv= rcv, scoring= "neg_mean_squared_error"))))
                self.ui.lineEdit_mse_lasreg.setText(las_mse)
                las_rmse = str(float('%.3f' % np.sqrt(np.mean(-cross_val_score(las_model,X_train,y_train, cv= rcv, scoring= "neg_mean_squared_error")))))
                self.ui.lineEdit_rmse_lasreg.setText(las_rmse)
                #Ridge
                rid_model = Ridge().fit(X_train,y_train)
                rid_mse = str(float('%.3f' % np.mean(-cross_val_score(rid_model,X_train,y_train,cv= rcv, scoring= "neg_mean_squared_error"))))
                self.ui.lineEdit_mse_ridreg.setText(rid_mse)
                rid_rmse = str(float('%.3f' % np.sqrt(np.mean(-cross_val_score(rid_model,X_train,y_train, cv= rcv, scoring= "neg_mean_squared_error")))))
                self.ui.lineEdit_rmse_ridreg.setText(rid_rmse)
                #ElasticNet
                ela_model = ElasticNet().fit(X_train,y_train)
                ela_mse = str(float('%.3f' % np.mean(-cross_val_score(ela_model,X_train,y_train,cv=rcv, scoring= "neg_mean_squared_error"))))
                self.ui.lineEdit_mse_elasreg.setText(ela_mse)
                ela_rmse = str(float('%.3f' % np.sqrt(np.mean(-cross_val_score(ela_model,X_train,y_train, cv= rcv, scoring= "neg_mean_squared_error")))))
                self.ui.lineEdit_rmse_elasreg.setText(ela_rmse)
                #coefficent
                model = LinearRegression().fit(X_train,y_train)
                coefficients = pd.concat([pd.DataFrame(X_train.columns),pd.DataFrame(np.transpose(model.coef_)).astype('float64')], axis = 1)
                coefficients.columns = ['Attributes','Coefficients']
                coefficients.Coefficients=coefficients.Coefficients.round(4)
                model_coe = PandasModel(coefficients)
                self.ui.tableView_reg_imp.setModel(model_coe)
                #maximum_coef
                mx = coefficients[coefficients.Coefficients == coefficients.Coefficients.max()]
                li = mx['Attributes'].tolist()
                att = str(li[0])
                cols = df.columns.tolist()
                lastone = str(cols[-1])
                self.ui.lineEdit_max_coef.setText(att)
                #scatter
                self.ui.MplWidget_scat_reg.canvas.axes.clear()
                self.ui.MplWidget_scat_reg.canvas.axes.scatter(x=att,y=lastone,data=df,color ='#c553a0')
                self.ui.MplWidget_scat_reg.canvas.axes.legend(('Max_Coef','Regressand'),loc='upper right')
                self.ui.MplWidget_scat_reg.canvas.axes.set_title('Regression ScatterPlot',fontsize=10,weight='bold')
                self.ui.MplWidget_scat_reg.canvas.axes.set_xlabel('Max_Coef')
                self.ui.MplWidget_scat_reg.canvas.axes.set_ylabel('Last_Column')
                self.ui.MplWidget_scat_reg.canvas.draw()
    
    def excelimport(self):
        rcv = int(self.ui.lineEdit_reg_cv.text())
        fileNameTuple = QFileDialog.getOpenFileName(self, 'OpenFile',"", "Excel (*.xls *.xlsx)")
        fileName = fileNameTuple[0]
        if fileName != "":
            df = pd.read_excel(fileName)
            modeldis = PandasModel(df)
            self.ui.tableView_reg_arff.setModel(modeldis)
            cols = df.columns.tolist()
            lastone = str(cols[-1])
            lastonetype = df.dtypes[lastone]
            pct = df.pivot_table(index=[lastone],aggfunc='size')
            df_co = pct.reset_index()
            df_co.columns = ['class','counts']
            if (lastonetype == np.object and len(df_co)<15) or (lastonetype == np.float64 and len(df_co)<15): #tested with clas.datasets optimum threshold '10' for the counts of classes
                QMessageBox.information(self, "Info", 'Your dataset is Classification!')
                df[lastone] = df[lastone].astype('category')
                df['encoded_labels'] = df[lastone].cat.codes #last column encoded 
                y = df.iloc[:,-1:]
                X= df.drop(df.iloc[:,-2:],axis=1)
                cat_fea_mask = X.dtypes==np.object
                ##for categorical attributes
                categorical_cols = X.columns[cat_fea_mask].tolist()
                le = LabelEncoder()
                X[categorical_cols] = X[categorical_cols].apply(lambda col: le.fit_transform(col))
                X_ = X.drop(X.columns[cat_fea_mask],axis=1)
                X_ = X_.apply(lambda x: x.fillna(int(x.mean())),axis= 0)
                X = pd.concat([X[categorical_cols],X_],axis=1)
                X = X.reindex(sorted(X.columns), axis=1)
                ##split
                X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
                ##decisiontree
                cart_model = DecisionTreeClassifier().fit(X_train,y_train)
                cart = float('%.3f' % cross_val_score(cart_model,X_test,y_test,cv=int(rcv)).mean())*100
                cart_str = str(round(cart,2))
                self.ui.lineEdit_desctree_5.setText(cart_str)
                ##lojistic
                loj_model = LogisticRegression(solver="liblinear").fit(X_train,y_train)
                loj = float('%.3f' % cross_val_score(loj_model,X_test,y_test,cv=int(rcv)).mean())*100
                loj_str = str(round(loj,2))
                self.ui.lineEdit_lojreg_5.setText(loj_str)
                ##Kneighbours
                knn_model = KNeighborsClassifier(n_neighbors = 2).fit(X_train, y_train)
                knn = float('%.3f' % cross_val_score(knn_model,X_test,y_test,cv=int(rcv)).mean())*100
                knn_str = str(round(knn,2))
                self.ui.lineEdit_kneigh_5.setText(knn_str)
                ##randomforest
                rfor_model = RandomForestClassifier().fit(X_train,y_train)
                rfor_pred = rfor_model.predict(X_test)
                rfor_acc = float('%.3f' % accuracy_score(y_test,rfor_pred))*100
                rfor_acc_str = str(round(rfor_acc,2))
                self.ui.lineEdit_random_5.setText(rfor_acc_str)
                ##LDA
                lda_model = LDA().fit(X_train,y_train)
                lda = float('%.3f' % cross_val_score(lda_model,X_test,y_test,cv=int(rcv)).mean())*100
                lda_str = str(round(lda,2))
                self.ui.lineEdit_mlp_5.setText(lda_str)
                #importance
                model = RandomForestClassifier().fit(X_train,y_train)
                dfi = pd.DataFrame({'Features':X_train.columns})
                dfi['Importance'] =(model.feature_importances_*100).round(4)
                dfi['Features'] = dfi['Features'].astype('category')
                dfi['Encoded_fea'] = dfi['Features'].cat.codes
                model_s = PandasModel(dfi)
                self.ui.tableView_class_imp.setModel(model_s)
                #max importance
                mx = dfi[dfi.Importance == dfi.Importance.max()]
                li = mx['Features'].tolist()
                fea = str(li[0])
                self.ui.lineEdit_max_imp.setText(fea)
                #bar
                self.ui.MplWidget_scat_clas.canvas.axes.clear()
                self.ui.MplWidget_scat_clas.canvas.axes.bar('Encoded_fea','Importance',data = dfi,color ='#4dc0d4')
                self.ui.MplWidget_scat_clas.canvas.axes.legend(('Importance','Features'),loc='upper right')
                self.ui.MplWidget_scat_clas.canvas.axes.set_title('Classification BarPlot',fontsize=10,weight='bold')
                self.ui.MplWidget_scat_clas.canvas.axes.set_xlabel('Encoded_fea')
                self.ui.MplWidget_scat_clas.canvas.axes.set_ylabel('Importance')
                self.ui.MplWidget_scat_clas.canvas.draw()
                if (lastonetype == np.float64 and lastonetype != np.object and len(df_co)>15):
                    QMessageBox.information(self, "Info", 'Your dataset is Regression!')
                    y = df.iloc[:,-1:]
                    X= df.drop(df.iloc[:,-1:],axis=1)
                    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
                    #linearreg
                    lin_model = LinearRegression().fit(X_train,y_train)
                    lin_mse = str(float('%.3f' % np.mean(-cross_val_score(lin_model,X_train,y_train,cv= int(rcv), scoring= "neg_mean_squared_error"))))
                    self.ui.lineEdit_mse_linreg.setText(lin_mse)
                    lin_rmse = str(float('%.3f' % np.sqrt(np.mean(-cross_val_score(lin_model,X_train,y_train, cv= int(rcv), scoring= "neg_mean_squared_error")))))
                    self.ui.lineEdit_rmse_linreg.setText(lin_rmse)
                    #LassoReg
                    las_model = Lasso().fit(X_train,y_train)
                    las_mse = str(float('%.3f' % np.mean(-cross_val_score(las_model,X_train,y_train,cv= int(rcv), scoring= "neg_mean_squared_error"))))
                    self.ui.lineEdit_mse_lasreg.setText(las_mse)
                    las_rmse = str(float('%.3f' % np.sqrt(np.mean(-cross_val_score(las_model,X_train,y_train, cv= int(rcv), scoring= "neg_mean_squared_error")))))
                    self.ui.lineEdit_rmse_lasreg.setText(las_rmse)
                    #Ridge
                    rid_model = Ridge().fit(X_train,y_train)
                    rid_mse = str(float('%.3f' % np.mean(-cross_val_score(rid_model,X_train,y_train,cv= int(rcv), scoring= "neg_mean_squared_error"))))
                    self.ui.lineEdit_mse_ridreg.setText(rid_mse)
                    rid_rmse = str(float('%.3f' % np.sqrt(np.mean(-cross_val_score(rid_model,X_train,y_train,cv= int(rcv), scoring= "neg_mean_squared_error")))))
                    self.ui.lineEdit_rmse_ridreg.setText(rid_rmse)
                    #ElasticNet
                    ela_model = ElasticNet().fit(X_train,y_train)
                    ela_mse = str(float('%.3f' % np.mean(-cross_val_score(ela_model,X_train,y_train,cv=int(rcv), scoring= "neg_mean_squared_error"))))
                    self.ui.lineEdit_mse_elasreg.setText(ela_mse)
                    ela_rmse = str(float('%.3f' % np.sqrt(np.mean(-cross_val_score(ela_model,X_train,y_train, cv=int(rcv), scoring= "neg_mean_squared_error")))))
                    self.ui.lineEdit_rmse_elasreg.setText(ela_rmse)
                    #coefficent
                    model = LinearRegression().fit(X_train,y_train)
                    coefficients = pd.concat([pd.DataFrame(X_train.columns),pd.DataFrame(np.transpose(model.coef_)).astype('float64')], axis = 1)
                    coefficients.columns = ['Attributes','Coefficients']
                    coefficients.Coefficients=coefficients.Coefficients.round(4)
                    model_coe = PandasModel(coefficients)
                    self.ui.tableView_reg_imp.setModel(model_coe)
                    #maximum_coef
                    mx = coefficients[coefficients.Coefficients == coefficients.Coefficients.max()]
                    li = mx['Attributes'].tolist()
                    att = str(li[0])
                    cols = df.columns.tolist()
                    lastone = str(cols[-1])
                    self.ui.lineEdit_max_coef.setText(att)
                    #scatter
                    self.ui.MplWidget_scat_reg.canvas.axes.clear()
                    self.ui.MplWidget_scat_reg.canvas.axes.scatter(x=att,y=lastone,data=df,color ='#c553a0')
                    self.ui.MplWidget_scat_reg.canvas.axes.legend(('Max_Coef','Regressand'),loc='upper right')
                    self.ui.MplWidget_scat_reg.canvas.axes.set_title('Regression ScatterPlot',fontsize=10,weight='bold')
                    self.ui.MplWidget_scat_reg.canvas.axes.set_xlabel('Max_Coef')
                    self.ui.MplWidget_scat_reg.canvas.axes.set_ylabel('Last_Column')
                    self.ui.MplWidget_scat_reg.canvas.draw()
                    
    def clearclass(self):
        self.ui.lineEdit_desctree_5.setText('')
        self.ui.lineEdit_lojreg_5.setText('')
        self.ui.lineEdit_kneigh_5.setText('')
        self.ui.lineEdit_random_5.setText('')
        self.ui.lineEdit_mlp_5.setText('')
        self.ui.lineEdit_max_imp.setText('')
        df= pd.DataFrame()
        model = PandasModel(df)
        self.ui.tableView_class_imp.setModel(model)
        self.ui.MplWidget_scat_clas.canvas.axes.clear()
        self.ui.MplWidget_scat_clas.canvas.draw()
        df= pd.DataFrame()
        model = PandasModel(df)
        self.ui.tableView_reg_arff.setModel(model)
    
    def clearregres(self):
        self.ui.lineEdit_mse_linreg.setText('')
        self.ui.lineEdit_rmse_linreg.setText('')
        self.ui.lineEdit_mse_ridreg.setText('')
        self.ui.lineEdit_rmse_ridreg.setText('')
        self.ui.lineEdit_mse_lasreg.setText('')
        self.ui.lineEdit_rmse_lasreg.setText('')
        self.ui.lineEdit_mse_elasreg.setText('')
        self.ui.lineEdit_mse_elasreg.setText('')
        self.ui.lineEdit_rmse_elasreg.setText('')
        self.ui.lineEdit_max_coef.setText('')
        df= pd.DataFrame()
        model = PandasModel(df)
        self.ui.tableView_reg_imp.setModel(model)
        self.ui.MplWidget_scat_reg.canvas.axes.clear()
        self.ui.MplWidget_scat_reg.canvas.draw()
        df= pd.DataFrame()
        model = PandasModel(df)
        self.ui.tableView_reg_arff.setModel(model)
        
    
    
    

    
        
        
        
        
        
        
