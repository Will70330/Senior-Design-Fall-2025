import sys
import os

current_dir = os.getcwd()
venv_site_packages = os.path.join(
    current_dir, 
    'venv', 
    'lib', 
    f'python{sys.version_info.major}.{sys.version_info.minor}', 
    'site-packages'
)
if os.path.exists(venv_site_packages) and venv_site_packages not in sys.path:
    sys.path.append(venv_site_packages)

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QSplitter
import pyvista as pv
from pyvistaqt import QtInteractor

print("PyVista imported")

app = QApplication(sys.argv)
win = QMainWindow()
splitter = QSplitter()
win.setCentralWidget(splitter)

plotter = QtInteractor(splitter)
print("QtInteractor created")
splitter.addWidget(plotter.interactor)

win.show()
print("Window shown")

from PyQt5.QtCore import QTimer
QTimer.singleShot(1000, app.quit)
sys.exit(app.exec_())