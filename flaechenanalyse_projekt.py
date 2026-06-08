from random import randint
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import numpy as np
from predict_final import KI
from drohne import DrohneThread
import cv2

w = 7
h = 7
options = ["strasse","feld","park","wald","wiese"]
S = 100
FPS = 120

# Stylesheet ---------------------------------------------

# Font families: Comfortaa, Helvetica/Helvetica Neue, Scientifica;

# interesting :) : border-color: #7DA53B;

APP_STYLE = """
* {
  background-color: #381806;

  font-family: Comfortaa;
  font-size: 15px;
  font-weight: 500;
}

QPushButton, QLineEdit { 
  color: #F0F2F3;
  background-color: #588E01;

  border-color: #3B6000;
  border-style: solid;
  border-width: 2px;
  border-radius: 10px;

  padding: 8px
}

QLineEdit:hover, QPushButton:hover {
  color: #F4D5AE;
  background-color: #709E25
}

QPushButton:pressed {
  color: #F4D5AE;
  background-color: #8BB743
}

QLabel#start_label {
    color: #F4D5AE;
    font-size: 65px;
    font-weight: 700;
}

QLabel#drone_control_hint_label {
    color: #4D5443;
    font-size: 11px
}



QTabBar {
    background: #84391C;
}
 QTabBar::tab {
    background: #84391C;
    color: #FCE9D1;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 1.5px;
}
 QTabBar::tab:selected {
     background: #964A2D;
     color: #F0F2F3;
}
QTabBar::tab:hover:!selected {
     background: #8C3E20;
     color: #EAEAEA;
}


"""
# --------------------------------------------------------

class Worker(QRunnable):
    """Worker thread."""
    @pyqtSlot()
    def run(self):
        """Your long-running job goes in this method."""
        print("Thread start")
        drohne.run()
        print("Thread complete")

class TabArrowFilter(QObject):
    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.main_window = main_window

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Left, Qt.Key_Right):
                self.main_window.keyPressEvent(event)  # forward to main window
                return True  # block tab switching
        return super().eventFilter(obj, event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.map = []
        self.tiles = []
        self.object = []
        
        self.setWindowTitle("Test")
        self.setGeometry(200, 200, 900, 870)

        # Create a Pygame surface and pass it to a QWindow
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        drohne.frame_updated.connect(self.update_frame)

        # Layout für Drohne Tab
        drohne_layout = QVBoxLayout()
        drohne_layout.addWidget(self.video_label)


        # Add the start and stop buttons to the main window
        button_layout = QHBoxLayout()

        hint_layout = QHBoxLayout()
        hint = QLabel("  T  󰗕   ·   L  󰗔   ·   F  󰄀   ·   WASD / Pfeiltasten  󰊴  ")
        hint.setObjectName("drone_control_hint_label")
        hint_layout.addWidget(hint)

        self.ip_input = QLineEdit()
        self.ip_input.setText(drohne.ip)
        self.ip_input.setMaxLength(15)
        button_layout.addWidget(self.ip_input)

        self.start_button = QPushButton("connect", self)
        self.start_button.clicked.connect(self.connect)

        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("disconnect", self)
        self.stop_button.clicked.connect(self.disconnect)
        button_layout.addWidget(self.stop_button)

        drohne_layout.addLayout(hint_layout)
        drohne_layout.addLayout(button_layout)
        

        drohne_widget = QWidget(self)
        drohne_widget.setLayout(drohne_layout)

        
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.West)
        self.tabs.setMovable(False)
        self.tab_arrow_filter = TabArrowFilter(self.tabs, self)
        self.tabs.tabBar().installEventFilter(self.tab_arrow_filter)
        
        
        start_wid = QWidget(self)
        start_lay = QVBoxLayout(start_wid)
        start_lay.setAlignment(Qt.AlignCenter)
        
        start_wid.setFixedSize(QSize(960,720))
        
        self.start_txt = QLabel("Fly\nThe Sky", start_wid)
        self.start_txt.setAlignment(Qt.AlignCenter)
        self.start_txt.setObjectName("start_label")

        self.under_start_txt = QLabel("KI & Drohne - Schulprojekt")
        self.under_start_txt.setAlignment(Qt.AlignCenter)
        self.under_start_txt.setObjectName("under_start_label")
        
        start_lay.addWidget(self.start_txt)
        start_lay.addWidget(self.under_start_txt)
        

        self.tabs.addTab(start_wid, "Start")       
        
        self.tabs.addTab(drohne_widget, "Drohne")
        
        fap = QWidget(self)
        ki_layout = QVBoxLayout()
        
        self.ki_button = QPushButton("Start KI", self)
        self.ki_button.clicked.connect(self.start_Ki)
        
        ki_layout.addWidget(self.ki_button) 
        
        fap.setLayout(ki_layout)

        
        self.tabs.addTab(fap, "KI")
        
        self.setCentralWidget(self.tabs)

        self.threadpool = QThreadPool()
        
        self.control_timer = QTimer()
        self.control_timer.timeout.connect(drohne.update)
        self.control_timer.start(1000 // FPS)

    @pyqtSlot(np.ndarray)
    def update_frame(self, frame):
        """Converts OpenCV BGR matrix into a Qt QPixmap and redraws it."""
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        
        # Convert BGR (OpenCV standard) to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Wrap OpenCV array inside QImage
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Display image while keeping aspect ratio scalability
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def get_ip(self):
        return self.ip_input.text()

    def connect(self):
        self.ip_input.setReadOnly(True)
        ip = self.get_ip()
        if not (ip == "192.168.0.101" or ip == "192.168.0.102"):
            print("falsche IP")
            return
        drohne.ip = ip
        drohne.fake_init()
        worker = Worker()
        self.threadpool.start(worker)

    def disconnect(self):
        #drohne.tello.end()
        pass

    def start_Ki(self):
        self.predict_test()
        self.verarbeitung_draw()
        self.map_draw()
        #pass

    def img_to_pie(self, fn, wedge, xy, zoom=1, ax = None):
        if ax==None: ax=plt.gca()
        im = plt.imread(fn, format='png')
        path = wedge.get_path()
        patch = PathPatch(path, facecolor='none')
        ax.add_patch(patch)
        imagebox = OffsetImage(im, zoom=zoom, clip_path=patch, zorder=-10)
        ab = AnnotationBbox(imagebox, xy, xycoords='data', pad=0, frameon=False)
        ax.add_artist(ab)

    def predict_test(self):
        ki = KI(model_path="best_v9.pt")

        ki.model_predict()

        self.map = ki.get_tile_map()
        self.tiles = ki.return_tiles_type_percent()
        self.object = ki.return_object_type_count()
        print(f"MAP: {ki.get_tile_map()}")
        print(f"TILES TYPE PERCENT: {ki.return_tiles_type_percent()}")
        print(f"OBJECT TYPE COUNT: {ki.return_object_type_count()}")

    def map_draw(self):
        layout = QGridLayout()

        print(self.map)
        for y, row in enumerate(self.map):
            for x, field,  in enumerate(row):
                if field in ["Strasse","Feld","See","Wald","Wiese","Fluss"]:
                    widget = QLabel()
                    widget.setPixmap(QPixmap(f"tile/{field}.png"))
                    widget.setScaledContents(True)
                    layout.addWidget(widget, y, x)
                    
        widget = QWidget(self)
        widget.setLayout(layout)
        
        self.tabs.addTab(widget, "Auswertung")

    def verarbeitung_draw(self):
        lab = QWidget(self)

        lay_ver = QGridLayout(lab)

        fig, ax = plt.subplots(figsize=(6, 6))
        ca_ver = FigureCanvas(fig)

        ca_ver.setFixedSize(QSize(600, 600))

        lay_ver.addWidget(ca_ver,0,0)

        lay_ver.setRowStretch(1, 1)     
        lay_ver.setColumnStretch(1, 1)

        #tiles = ["Wald", "Fluss", "Strasse", "Feld", "Wiese", "See"]
        tiles = []
        values = []
        for i in self.tiles:
            tiles.append(i[0])
            values.append(i[1])

        plt.title("Flächenanalyse")
        plt.gca().axis("equal")
        wedges, texts = plt.pie(values, startangle=90, labels=tiles,
                                wedgeprops = { 'linewidth': 2, "edgecolor" :"k","fill":False, }, textprops = { 'color': '#C7C8C9', 'size': 12})

        positions = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
        #zooms = [1,1,3.5,0.5,0.5,1]
        zooms = []
        for tile in tiles:
            if tile == "Strasse":
                zooms.append(3.5)
            elif tile == "Feld":
                zooms.append(0.5)
            elif tile == "Wiese":
                zooms.append(0.5)
            else:
                zooms.append(1)

        for i in range(len(tiles)):
            fn = "img/{}.png".format(tiles[i].lower())
            self.img_to_pie(fn, wedges[i], positions[i], zooms[i] )
            wedges[i].set_zorder(10)


        x = 0.5 + np.arange(5)
        y = [1, 5, 7, 2, 1]
        biggest = max(y)

        '''   
        objekt = []
        y = []
        for i in self.object:
            objekt.append(i[0])
            y.append(i[1])

        x = 0.5 + np.arange(len(objekt))
        biggest = max(y)
        ''' 

        # plot
        fig, ax = plt.subplots()

        colors = ["purple", "red", "blue", "green", "yellow"]
        ax.bar(x, y, width=0.8, color=colors, edgecolor="white", linewidth=0.7)

        ax.set(xlim=(0, 5), xticks=np.arange(0,0),
            ylim=(0, biggest + 1), yticks=np.arange(1, biggest + 1))

        kategorien = ["Yippie", "Haus", "Fahrzeug", "AKW", "Wohnkomplex"]

        ax.set_xticks(x)
        ax.set_xticklabels(kategorien)
        #ax.legend(title)

        canvas_bar = FigureCanvas(plt.gcf())
        canvas_bar.draw()

        lay_ver.addWidget(canvas_bar, 0, 1)
            
            
        ca_ver.draw()
        self.tabs.addTab(lab, "Verarbeitung")
                    
    def keyPressEvent(self, key):
        drohne.keydown(key)
    
    def keyReleaseEvent(self, key):
        drohne.keyup(key)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setStyleSheet(APP_STYLE)

    drohne = DrohneThread()
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
