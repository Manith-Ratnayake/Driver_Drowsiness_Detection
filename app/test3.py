
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class PlotWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Create a Figure
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Draw initial plot
        self.plot_initial()

    def plot_initial(self):
        ax = self.figure.add_subplot(111)
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y, label="Sine wave")
        ax.set_title("Example Plot")
        ax.legend()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = PlotWidget()
    w.show()
    sys.exit(app.exec_())
