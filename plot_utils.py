import matplotlib.pyplot as plt
import numpy as np

class Plotter(object):
    def __init__(self, rows=4, columns=4):
        fig=plt.figure(figsize=(8, 8))
        idx = 0

        plots = []

        for i in range(1, columns*rows +1):
            img = np.zeros((10, 10, 3))
            ax = fig.add_subplot(rows, columns, i)
            ax.axis('off')

            new_plot = plt.imshow(img, extent=[0, 1, 0, 1])
            plots.append(new_plot)

            idx += 1

        plt.show(block=False)

        plt.pause(.1)
        plt.ion()

        self.fig = fig
        self.plots = plots

    def plot(self, images):
        k = min(len(images), len(self.plots))

        for i in range(k):
            img_data = images[i]
            shape = np.shape(img_data)

            if len(shape) == 3 and shape[2] == 1:
                img_data = np.tile(img_data, (1, 1, 3))

            self.plots[i].set_data(img_data)

        self.fig.canvas.draw()
        plt.pause(.1)
