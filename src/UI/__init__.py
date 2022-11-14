from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from model.RBFN import RBFN
from PyQt5 import QtCore, QtWidgets
from math import sin, cos, asin, tan, pi
from typing import Callable, Union
from UI.pyFile.mainPage import Ui_Form
import matplotlib
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
matplotlib.use('Qt5Agg')


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class mainPageWidget(QtWidgets.QWidget, Ui_Form):
    def __init__(self, readData: Callable[[str], list[str]], runData: Callable[[list[str], list[str]], RBFN]):
        super(QtWidgets.QWidget, self).__init__()
        self.setupUi(self)
        self.model: Union[RBFN, None] = None
        self.currentData: list[float] = [0, 0, 90]
        self.chart: FigureCanvasQTAgg = MplCanvas(
            self, width=5, height=4, dpi=100)
        self.ChartLayout.addWidget(self.chart)
        self.inputDim = 0
        self.timer=None
        def process():
            track = readData('resource/track.txt')
            if self.radioButton_2.isChecked():
                dataset = readData('resource/train4DAll.txt')
                self.pushButton.setText('training')
            elif self.radioButton.isChecked():
                dataset = readData('resource/train6DAll.txt')
                self.pushButton.setText('data training')
            else:
                self.pushButton.setText('not selected')
                return
            self.inputDim = len(dataset[0].split())-1
            flTrack: list[list[float]] = [
                list(map(lambda x:float(x), i.split(','))) for i in track]
            self.currentData = flTrack[0]
            self.trackInfo = flTrack
            self.drawNewTrack()
            self.trackList.clear()
            self.trackList.addItem('track data record')
            def runAndUpdate() -> None:
                self.model = runData(track, dataset)
                self.modelChanged()
                self.pushButton.setText('start training')
            QtCore.QTimer.singleShot(50, runAndUpdate)

        self.pushButton.clicked.connect(process)
        self.simulateBut.clicked.connect(self.simulate)

    def drawNewTrack(self):
        trackInfo=self.trackInfo
        self.chart.axes.cla()
        startP=[min(trackInfo[2][0], trackInfo[1][0]), min(trackInfo[2][1], trackInfo[1][1])]
        endP=[max(trackInfo[2][0], trackInfo[1][0]), max(trackInfo[2][1], trackInfo[1][1])]
        self.chart.axes.add_patch(Rectangle(startP, endP[0]-startP[0], endP[1]-startP[1], facecolor = 'orange', lw=0))
        self.chart.axes.plot([-6, 6], [0, 0], 'g')
        self.currentWall: list[list[float]] =\
            [[[trackInfo[i][0], trackInfo[i+1][0]], [trackInfo[i][1], trackInfo[i+1][1]]]
                for i in range(3, len(trackInfo)-1)]
        for wall in self.currentWall:
            self.chart.axes.plot(wall[0], wall[1], 'g')
        self.chart.draw()

    def modelChanged(self):
        if self.model == None:
            self.simulateBut.setText('model not ready, train first')
            self.simulateBut.setEnabled(False)
            return
        self.simulateBut.setText('model ready, start simulate')
        self.simulateBut.setEnabled(True)

    def simulate(self):
        if self.timer is not None: self.timer.stop()
        self.wallLine=[]
        for wall in self.currentWall:
            dy, dx = wall[1][1]-wall[1][0], wall[0][1]-wall[0][0] #dy*x-dx*y=c
            c = dy*wall[0][0]-dx*wall[1][0]
            self.wallLine.append([dy, dx, c, wall[0], wall[1]])
        self.x, self.y, self.phy=self.currentData[0], self.currentData[1], self.currentData[2]
        self.xs, self.ys=[], []
        self.theta, self.counter=0, 0
        def timerDo():
            x, y, phy, theta, counter, wallLine=self.x, self.y, self.phy, self.theta, self.counter, self.wallLine
            self.xs.append(x)
            self.ys.append(y)
            self.drawNewTrack()
            self.chart.axes.plot(self.xs, self.ys, marker='o', markersize=3, color="red")
            self.chart.axes.add_patch(plt.Circle((x, y), 3, color='b', fill=False))
            self.chart.draw()
            counter+=1
            trackInfo=self.trackInfo
            def dis(x, y, endline):
                a, b=endline[0][1]-endline[1][1], endline[1][0]-endline[0][0]
                c=-a*endline[0][0]-b*endline[0][1]
                return abs(a*x+b*y+c)/np.sqrt(a*a+b*b)
            start=[min(trackInfo[1][0], trackInfo[2][0]), min(trackInfo[1][1], trackInfo[2][1])]
            end=[max(trackInfo[1][0], trackInfo[2][0]), min(trackInfo[1][1], trackInfo[2][1])]
            if dis(x, y, (start, end))<=3:self.timer.stop()
            ldis, rdis, updis=3000, 3000, 3000
            phy = (phy+450)%360-90
            upvx, upvy=cos(phy*pi/180), sin(phy*pi/180) #line = x+upvx*t, y+upvy*t -> dy*x-dx*y+t(upvx*dy-upvy*dx)=c
            rvx, rvy=cos((phy-45)*pi/180), sin((phy-45)*pi/180)
            lvx, lvy=cos((phy+45)*pi/180), sin((phy+45)*pi/180)
            for wall in wallLine:
                if wall[0]*upvx!=wall[1]*upvy:
                    t=-(wall[0]*x-wall[1]*y-wall[2])/(wall[0]*upvx-wall[1]*upvy)
                    if t>=0: 
                        xintersect, yintersect = x+upvx*t, y+upvy*t
                        if ((xintersect-wall[3][0])*(xintersect-wall[3][1])<=0 or \
                        abs(xintersect-wall[3][0])<0.01 or abs(xintersect-wall[3][1])<0.01) and\
                            ((yintersect-wall[4][0])*(yintersect-wall[4][1])<=0 or \
                        abs(yintersect-wall[4][0])<0.01 or abs(xintersect-wall[4][1])<0.01):
                            updis=min(updis, t*math.sqrt(upvx*upvx+upvy*upvy))
                if wall[0]*rvx!=wall[1]*rvy:
                    t=-(wall[0]*x-wall[1]*y-wall[2])/(wall[0]*rvx-wall[1]*rvy)
                    if t>=0: 
                        xintersect, yintersect=x+rvx*t, y+rvy*t
                        if ((xintersect-wall[3][0])*(xintersect-wall[3][1])<=0 or \
                        abs(xintersect-wall[3][0])<0.01 or abs(xintersect-wall[3][1])<0.01) and\
                            ((yintersect-wall[4][0])*(yintersect-wall[4][1])<=0 or \
                        abs(yintersect-wall[4][0])<0.01 or abs(xintersect-wall[4][1])<0.01):
                            rdis=min(rdis, t*math.sqrt(rvx*rvx+rvy*rvy))
                if wall[0]*lvx!=wall[1]*lvy:
                    t=-(wall[0]*x-wall[1]*y-wall[2])/(wall[0]*lvx-wall[1]*lvy)
                    if t>=0: 
                        xintersect, yintersect=x+lvx*t, y+lvy*t
                        if ((xintersect-wall[3][0])*(xintersect-wall[3][1])<=0 or \
                        abs(xintersect-wall[3][0])<0.01 or abs(xintersect-wall[3][1])<0.01) and\
                            ((yintersect-wall[4][0])*(yintersect-wall[4][1])<=0 or \
                        abs(yintersect-wall[4][0])<0.01 or abs(xintersect-wall[4][1])<0.01):
                            ldis=min(ldis, t*math.sqrt(lvx*lvx+lvy*lvy))
            if updis==3000:
                updis=20
            if rdis==3000:
                rdis=20
            if ldis==3000:
                ldis=20
            if self.inputDim==3:
                theta = self.model.predict(np.array([[updis-6, rdis-6, ldis-6]]))[0][0]
            else:
                theta = self.model.predict(np.array([[x, y, updis-6, rdis-6, ldis-6]]))[0][0]
            self.trackList.addItem(f'{x:.2f}\t {y:.2f}\t {updis:.2f}\t {rdis:.2f}\t {ldis:.2f}\t {phy:.2f}\t {theta:.2f}')
            #f.write(f'{x:.7f} {y:.7f} {updis:.7f} {rdis:.7f} {ldis:.7f} {theta:.7f}\n')
            x += cos(phy*pi/180+theta*pi/180)+sin(phy*pi/180)*sin(theta*pi/180)
            y += sin(phy*pi/180+theta*pi/180)-sin(theta*pi/180)*cos(phy*pi/180)
            phy-=asin(2*sin(theta*pi/180)/6)*180/pi
            self.x, self.y, self.phy, self.theta, self.counter=x, y, phy, theta, counter
        from PyQt5.QtCore import QTimer
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(timerDo)
        self.timer.start()
        
                

                    

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, readData: Callable[[str], list[str]], runData: Callable[[list[str], list[str]], RBFN]):
        super(MainWindow, self).__init__()
        mp: QtWidgets.QWidget = mainPageWidget(readData, runData)
        self.setCentralWidget(mp)
        self.show()


class App():
    def __init__(self, readData: Callable[[str], list[str]], runData: Callable[[list[str], list[str]], RBFN], *args, **kwargs):
        self.readData: Callable[[str], list[str]] = readData
        self.runData: Callable[[list[str], list[str], RBFN]] = runData

    def startAppSync(self):
        import sys
        app = QtWidgets.QApplication(sys.argv)
        w = MainWindow(self.readData, self.runData)
        app.exec()
