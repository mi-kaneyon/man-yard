from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt, QRectF, QPointF, QTimer
from PyQt5.QtGui import QPainter, QColor, QPen, QPolygonF
import sys
import random
import math

class DrawingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.last_point = None

    def initUI(self):
        self.setWindowTitle('Line DRAW')
        self.setGeometry(100, 100, 800, 600)

        self.example_view = QGraphicsView()
        self.example_scene = QGraphicsScene()
        self.example_view.setScene(self.example_scene)
        self.example_view.setRenderHint(QPainter.Antialiasing)
        self.example_view.setSceneRect(QRectF(0, 0, 400, 400))
        self.example_view.setFixedSize(400, 400)

        self.user_view = QGraphicsView()
        self.user_scene = QGraphicsScene()
        self.user_view.setScene(self.user_scene)
        self.user_view.setRenderHint(QPainter.Antialiasing)
        self.user_view.setSceneRect(QRectF(0, 0, 400, 400))
        self.user_view.setFixedSize(400, 400)
        self.user_view.setInteractive(True)
        self.user_view.mousePressEvent = self.draw_point
        self.user_view.mouseMoveEvent = self.draw_line

        self.compare_btn = QPushButton('COMPARE', self)
        self.compare_btn.clicked.connect(self.compare_drawings)
        
        self.restart_btn = QPushButton('RESTART', self)
        self.restart_btn.clicked.connect(self.restart_app)

        self.exit_btn = QPushButton('EXIT', self)
        self.exit_btn.clicked.connect(self.close)

        self.result_label = QLabel("Result will be displayed here", self)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.compare_btn)
        btn_layout.addWidget(self.restart_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.example_view)
        layout.addWidget(self.user_view)
        layout.addLayout(btn_layout)
        layout.addWidget(self.result_label)
        layout.addWidget(self.exit_btn)

        container = QWidget()
        container.setLayout(layout)

        self.setCentralWidget(container)
        self.show()

        self.draw_example_shapes()
        self.draw_dots(self.example_scene)
        self.draw_dots(self.user_scene)

    def draw_dots(self, scene):
        pen = QPen(QColor(0, 0, 0))
        pen.setWidth(2)
        for i in range(0, 401, 50):
            for j in range(0, 401, 50):
                scene.addEllipse(i, j, 2, 2, pen)

    def draw_point(self, event):
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(2)
        point = event.pos()
        self.user_scene.addEllipse(point.x(), point.y(), 2, 2, pen)
        self.last_point = point

    def draw_line(self, event):
        if self.last_point:
            pen = QPen(QColor(255, 0, 0))
            pen.setWidth(2)
            new_point = event.pos()
            self.user_scene.addLine(self.last_point.x(), self.last_point.y(), new_point.x(), new_point.y(), pen)
            self.last_point = new_point

    def draw_example_shapes(self):
        pen = QPen(QColor(0, 0, 255))
        pen.setWidth(2)
        grid_points = [(i, j) for i in range(0, 401, 50) for j in range(0, 401, 50)]
        for _ in range(2):
            points = random.sample(grid_points, random.randint(3, 5))
            self.example_scene.addPolygon(QPolygonF([QPointF(x, y) for x, y in points]), pen)

    def compare_drawings(self):
        # Dummy logic for now
        result = "WIN!"
        self.result_label.setText(result)
        self.result_label.setStyleSheet('color: red')

    def restart_app(self):
        self.example_scene.clear()
        self.user_scene.clear()
        self.draw_example_shapes()
        self.draw_dots(self.example_scene)
        self.draw_dots(self.user_scene)
        self.result_label.setText("Result will be displayed here")
        self.result_label.setStyleSheet('color: black')

if __name__ == '__main__':
    app = QApplication([])
    ex = DrawingApp()
    sys.exit(app.exec_())
