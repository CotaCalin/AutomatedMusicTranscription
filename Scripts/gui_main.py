import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtMultimedia import QSound
from UI.ui_main import UIMain
from controller import Controller
from Utils.cfg_reader import CfgReader
from Model.model_factory import ModelFactory
from trainer import Trainer
from Preprocessing.preprocess import Preprocessor
import os

import threading
import time

DEFAULT_CFG = "../licenta.cfg"

class getPredictionThread(QtCore.QThread):
    completed = QtCore.pyqtSignal()

    def __init__(self, controller, path):
        QtCore.QThread.__init__(self)
        self.controller = controller
        self.path = path

    def __del__(self):
        self.wait()

    def run(self):
        self.controller.predict(self.path)
        self.completed.emit()

class PredictionPopup(QtWidgets.QWidget):
    def __init__(self, path):
        QtWidgets.QWidget.__init__(self)
        self.path = path
        self.setWindowTitle("Prediction for " + path)
        self.setWindowIcon(QtGui.QIcon('logo.png'))

        self.showOutput()

    def showPopup(self, message):
        QtWidgets.QMessageBox.about(self, "Hello there", message)

    def showOutput(self):
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignCenter)

        self.label = QtWidgets.QLabel(self)
        self.png = [os.path.join(self.path, x) for x in os.listdir(self.path) if x.startswith("predict_sheet")]
        print(self.png)

        self.i = 0
        self.pageCountLabel = QtWidgets.QLabel(self)
        self.pageCountLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.pageCountLabel.setText("{0}/{1}".format(self.i + 1, len(self.png)))

        self.pixMap = QtGui.QPixmap(self.png[0])
        self.label.setPixmap(self.pixMap)

        # page scrolling layout
        self.btnLayout = QtWidgets.QHBoxLayout(self)
        self.btn1 = QtWidgets.QPushButton("<", self)
        self.btn1.clicked.connect(self.goLeft)
        self.btn2 = QtWidgets.QPushButton(">", self)
        self.btn2.clicked.connect(self.goRight)

        self.btn1.resize(10, 10)
        self.btn2.resize(10, 10)

        # audio playing layout
        self.btnPlay = QtWidgets.QPushButton("Play", self)
        self.btnPlay.clicked.connect(self.playPredict)
        self.btnStop = QtWidgets.QPushButton("Stop", self)
        self.btnStop.clicked.connect(self.stopPredict)

        self.btnPlay.resize(10, 10)
        self.btnStop.resize(10, 10)

        self.audioLayout = QtWidgets.QHBoxLayout(self)

        self.audioLayout.setAlignment(QtCore.Qt.AlignCenter)
        self.audioW = QtWidgets.QWidget(self)
        self.audioLayout.addWidget(self.btnPlay)
        self.audioLayout.addWidget(self.btnStop)
        self.audioW.setLayout(self.audioLayout)

        self.btnLayout.setAlignment(QtCore.Qt.AlignCenter)
        self.w = QtWidgets.QWidget(self)
        self.btnLayout.addWidget(self.btn1)
        self.btnLayout.addWidget(self.btn2)
        self.w.setLayout(self.btnLayout)

        self.layout.addWidget(self.audioW)
        self.layout.addWidget(self.w)
        self.layout.addWidget(self.pageCountLabel)
        self.layout.addWidget(self.label)

        self.setLayout(self.layout)

        self.sound = QSound(os.path.join(self.path, "predict.wav"))
        self.show()

    def goLeft(self):
        if self.i == 0:
            self.showPopup("You're already at the first page")
            return

        self.i -= 1
        self.refreshSheet()


    def goRight(self):
        if self.i == len(self.png)-1:
            self.showPopup("You're already at the last page")
            return

        self.i += 1
        self.refreshSheet()

    def refreshSheet(self):
        self.pageCountLabel.setText("{0}/{1}".format(self.i + 1, len(self.png)))
        self.pixMap = QtGui.QPixmap(self.png[self.i])
        self.label.setPixmap(self.pixMap)

    def playPredict(self):
        self.sound.play()

    def stopPredict(self):
        self.sound.stop()


class Window(QtWidgets.QMainWindow):
    def __init__(self, controller):
        super(Window, self).__init__()
        self.setGeometry(500, 500, 500, 300)
        self.setWindowTitle("Theia7")
        self.setWindowIcon(QtGui.QIcon('logo.png'))
        self.__controller = controller

        self.midiSelected = False
        self.wid = QtWidgets.QWidget(self)
        self.setCentralWidget(self.wid)
        self.popup = None
        self.threads = []

        #x = threading.Thread(target=self.__controller.loadModel, args=())
        #self.threads.append(x)
        #x.start()
        #self.__controller.loadModel()
        exitAction = QtWidgets.QAction("&Exit", self)
        exitAction.setShortcut("ctrl+Q")
        exitAction.setStatusTip("Exit the application")
        exitAction.triggered.connect(self.onQuitBtnClicked)

        self.statusBar()
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu("&File")
        fileMenu.addAction(exitAction)

        self.home()

    def home(self):
        self.layout = QtWidgets.QVBoxLayout()
        titleFont = QtGui.QFont("Arial", 30)
        currentSongFont = QtGui.QFont("Arial", 15)

        self.label = QtWidgets.QLabel()
        self.label.setText("Theia7\n your personal music transcriber")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setFont(titleFont)

        self.songLabel = QtWidgets.QLabel()
        self.songLabel.setText("No song selected")
        self.songLabel.setAlignment(QtCore.Qt.AlignLeft)
        self.songLabel.setFont(currentSongFont)

        self.layout.addWidget(self.label)

        self.layout.addStretch()
        self.layout.addWidget(self.songLabel)

        self.progress = QtWidgets.QProgressBar(self)

        self.layout.addWidget(self.progress)

        #self.label.move(50, 50)

        icon = QtGui.QIcon('logo.png')

        btn = QtWidgets.QPushButton("Transpose", self)
        btn.clicked.connect(self.onPredictClicked)

        btn.setIcon(icon)

        self.layout.addWidget(btn)

        self.wid.setLayout(self.layout)

        predictAction = QtWidgets.QAction(QtGui.QIcon('logo.png'), "Predict", self)
        predictAction.triggered.connect(self.onPredictClicked)


        #self.progress.setGeometry(200, 80, 250, 20)

        c = 0

        self.progress.setValue(c)

        self.progress.hide()

        self.show()

    def onPredictClicked(self):
        QtWidgets.QMessageBox.about(self, "Prediction notification", "Select a valid .mid file")

        name = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File')
        #name = ["d:\\datasets\\licenta_demo\\BACH No-03.mid"]
        if not name[0].endswith(".mid"):
            QtWidgets.QMessageBox.about(self, "Bad file selected", "Please select a valid .mid file")
            return

        self.songLabel.setText("Transcribing song: " + name[0])
        self.pred_termination = threading.Event()

        try:
            x = getPredictionThread(self.__controller, name[0])
            x.completed.connect(self.onPredictionDone)
            self.threads.append(x)
            x.start()

            pass
        except Exception as e:
            QtWidgets.QMessageBox.about(self, "Exception encountered", str(e))

    def onPredictionDone(self):
        self.songLabel.setText("Prediction done: {0}".format(self.__controller.getLastPredOut()))
        self.popup = PredictionPopup(self.__controller.getLastPredOut())
        self.popup.setGeometry(QtCore.QRect(100, 100, 400, 200))
        self.popup.show()


    def onQuitBtnClicked(self):
        choice = QtWidgets.QMessageBox.question(self, "Leaver notice",
                                            "Are you sure you want to exit?",
                                            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

        if choice == QtWidgets.QMessageBox.No:
            return

        for thread in self.threads():
            thread.join()

        sys.exit(0)


def run():
    factory = ModelFactory()
    cfg = CfgReader(DEFAULT_CFG)
    pre = Preprocessor(cfg)
    trainer = Trainer(factory)
    cont = Controller(cfg, pre, trainer)

    app = QtWidgets.QApplication(sys.argv)
    gui = Window(cont)
    sys.exit(app.exec_())

run()
