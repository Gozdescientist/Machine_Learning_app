from PyQt5.QtWidgets import QDialog
from register import Ui_Dialog

class SecondPage(QDialog):
    
    def __init__(self):
        super.__init__()
        
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        
        

