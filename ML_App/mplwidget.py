# ------------------------------------------------------
# -------------------- mplwidget.py --------------------
# ------------------------------------------------------
from PyQt5.QtWidgets import QWidget,QVBoxLayout

from PyQt5.QtCore import QSize

from matplotlib.backends.backend_qt5agg import FigureCanvas

from matplotlib.figure import Figure

from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)


class MplWidget(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        self.canvas = FigureCanvas(Figure())
        toolbar = NavigationToolbar(self.canvas, self)
        toolbar.setIconSize(QSize(23, 23))

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        vertical_layout.addWidget(toolbar)
        

        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.setLayout(vertical_layout)