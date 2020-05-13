import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class digram_show:
    def show_scatter(x,y):
        plt.scatter(x,y)
        plt.show()
    def show_plot(x,y):
        plt.plot(x,y)
        plt.show()
    def show_plot_3d(x,y,z):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot(x, y, z, label='parametric curve')
        ax.legend()
        plt.show()
    def show_scatter_3d(x,y,z):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(x, y, z, label='parametric scatter')
        ax.legend()
        plt.show()
    def show_plot_diff(X,Y,x,y):
        plt.plot(X,Y, color='blue', linewidth=2)
        plt.plot(x,y, color='yellow', linewidth=2)
        plt.show()