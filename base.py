import math
import sys
import numpy as np
import numba as nb

from matplotlib.widgets import Slider
from matplotlib.backend_tools import ToolBase

from numba import cuda
from pylab import plt
from PIL import Image
from timeit import default_timer as timer
from tkinter import filedialog

import matplotlib
matplotlib.rcParams['savefig.dpi'] = 1000
matplotlib.rcParams['savefig.frameon'] = False
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams["toolbar"] = "toolmanager"


def create_image_array(kernel, xmin, xmax, ymin, ymax, max_iter, base_accuracy, splits=None, *args):
    if abs(xmax - xmin) > abs(ymax - ymin):
        ny = base_accuracy
        nx = int((base_accuracy * abs(xmax - xmin) / abs(ymax - ymin)))
    else:
        nx = base_accuracy
        ny = int(base_accuracy * abs(ymax - ymin) / abs(xmax - xmin))

    xstride = abs(xmax - xmin) / nx
    ystride = abs(ymax - ymin) / ny
    topleft = nb.complex128(xmin + 1j * ymax)
    image_array = np.zeros((ny, nx, 3), dtype=np.uint8)

    if splits:
        run_kernel_split(kernel, image_array, topleft, xstride, ystride, max_iter, splits, *args)
    else:
        run_kernel(kernel, image_array, topleft, xstride, ystride, max_iter, *args)

    return image_array


class Explorer:
    def __init__(self, kernel, xmin, xmax, ymin, ymax, base_accuracy, max_iter, interpolation='none', splits=None, *args):
        fig, ax = plt.subplots()

        self.base_accuracy = base_accuracy
        self.height = base_accuracy
        self.width = base_accuracy
        self.max_iter = max_iter
        self.kernel = kernel
        self.x = np.linspace(xmin, xmax, self.width)
        self.y = np.linspace(ymin, ymax, self.height)
        self.splits = splits
        self.args = args
        self.ax = ax
        self.fig = fig
        self.interpolation = interpolation
        self.image_array = None

        class ImageSaver(ToolBase):
            image_array = None
            description = 'Save the image only'

            def trigger(self, *args, **kwargs):
                path = filedialog.asksaveasfilename(initialfile='Fractal_1',
                                                    defaultextension='png',
                                                    filetypes=[('PNG', ".png")])
                image = Image.fromarray(self.image_array, mode='RGB')
                image.save(path, "PNG", quality=95, optimize=False)

        tm = fig.canvas.manager.toolmanager
        self.image_saver = tm.add_tool("Save Image", ImageSaver)
        fig.canvas.manager.toolbar.add_tool(tm.get_tool("Save Image"), "toolgroup")

        self.slider_box = plt.axes([0.12, 0.02, 0.7, 0.03])
        self.slider = Slider(self.slider_box, 'Max iter:', 100, 50000, valinit=max_iter, valstep=10)
        self.slider.set_val(max_iter)
        self.slider.on_changed(self.on_slider_change)
        plt.subplots_adjust(bottom=0.1, top=0.95)


    def show(self):
        self.ax.imshow(np.zeros((100, 100, 3), dtype=np.uint8),
                       origin='lower',
                       extent=(self.x.min(), self.x.max(), self.y.min(), self.y.max()),
                       interpolation=self.interpolation,
                       resample=True)
        self.ax.callbacks.connect('ylim_changed', self.draw)
        self.draw(self.ax)
        plt.show()

    def on_slider_change(self, *args):
        self.draw(self.ax)

    def draw(self, ax):
        ax.set_autoscale_on(False)
        dims = ax.get_window_extent().bounds

        self.width = int(dims[2] + 0.5)
        self.height = int(dims[2] + 0.5)

        xmin, ymin, xdelta, ydelta = ax.viewLim.bounds
        xmax = xmin + xdelta
        ymax = ymin + ydelta

        im = ax.images[-1]
        self.x = np.linspace(xmin, xmax, self.width)
        self.y = np.linspace(ymin, ymax, self.height)
        self.image_array = create_image_array(self.kernel, xmin, xmax, ymin,ymax,
                                              int(self.slider.val), self.base_accuracy,
                                              self.splits, *self.args)
        self.image_saver.image_array = self.image_array
        im.set_data(self.image_array)
        im.set_extent((xmin, xmax, ymax, ymin))
        ax.figure.canvas.draw_idle()

def create_image(kernel,
                 xmin, xmax, ymin, ymax,
                 max_iter,
                 base_accuracy,
                 path='fractal.png',
                 show=True,
                 splits=None,
                 *args):
    image_array = create_image_array(kernel, xmin, xmax, ymin,ymax, max_iter, base_accuracy, splits, *args)

    image = Image.fromarray(image_array, mode='RGB')
    image.save(path, "PNG", quality=95, optimize=True)
    if show: image.show()


def run_kernel(kernel, image, topleft, xstride, ystride, max_iter, *args):
    start = timer()

    dimage = cuda.to_device(image)
    threadsperblock = (32, 16)
    blockspergrid = (math.ceil(image.shape[0] / threadsperblock[0]), math.ceil(image.shape[1] / threadsperblock[1]))

    kernel[blockspergrid, threadsperblock](dimage, topleft, xstride, ystride, max_iter, *args)
    dimage.to_host()

    sys.stdout.write('\r' + "Fractal calculated on GPU in %f s" % (timer() - start))

def run_kernel_split(kernel, image, topleft, xstride, ystride, max_iter, splits, *args):
    start = timer()
    split = np.linspace(0, image.shape[0], splits + 1, dtype=np.uint32)
    threadsperblock = (32, 8)

    dimage = cuda.to_device(image)

    for n in range(len(split) - 1):
        sys.stdout.write('\r' + "Processing split [%s / %s]" % (n, len(split)-1))
        blockspergrid = (math.ceil((split[n + 1] - split[n]) / threadsperblock[0]),
                         math.ceil(image.shape[1] / threadsperblock[1]))
        kernel[blockspergrid, threadsperblock](dimage, topleft, xstride, ystride, max_iter,
                                                       split[n], split[n + 1],
                                                       *args)
        dimage.copy_to_host(image)

    sys.stdout.write('\r' + "Fractal calculated on GPU in %f s" % (timer() - start))


