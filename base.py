import math
import sys
import numpy as np
import numba as nb

from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider

from numba import cuda
from pylab import plt
from PIL import Image
from timeit import default_timer as timer

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

# Very similar to matplotlib's mandelbrot example
class Explorer:
    def __init__(self, kernel, xmin, xmax, ymin, ymax, base_accuracy, max_iter, interpolation='none', splits=None, *args):
        class UpdatingRect(Rectangle):
            def __call__(self, ax):
                self.set_bounds(*ax.viewLim.bounds)
                ax.figure.canvas.draw_idle()

        fig, ax = plt.subplots()

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
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
        self.interpolation = interpolation
        self.image_array = create_image_array(self.kernel, xmin, xmax, ymin, ymax,
                                              self.max_iter, self.base_accuracy,
                                              self.splits, *self.args)

        sliderbox = plt.axes([0.15, 0.05, 0.65, 0.03])
        slider = Slider(sliderbox, 'Max iter', 100, 50000, valinit=max_iter, valstep=10)
        slider.set_val(max_iter)
        slider.on_changed(self.update_max_iter)
        self.ax.imshow(self.image_array,
                       origin='lower',
                       extent=(self.x.min(), self.x.max(), self.y.min(), self.y.max()),
                       interpolation=self.interpolation,
                       resample=True)

        plt.subplots_adjust(bottom=0.2, top=0.95)

        rect = UpdatingRect((0, 0), 0, 0, facecolor='None', edgecolor='black', linewidth=1.0)
        rect.set_bounds(ax.viewLim.bounds)
        ax.add_patch(rect)

        ax.callbacks.connect('ylim_changed', rect)
        ax.callbacks.connect('ylim_changed', self.draw)

        plt.show()

    def update_max_iter(self, val):
        self.max_iter = int(val)
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
                                              self.max_iter, self.base_accuracy,
                                              self.splits, *self.args)
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


def run_kernel(f_kernel, image, topleft, xstride, ystride, max_iter, *args):
    start = timer()

    dimage = cuda.to_device(image)
    threadsperblock = (32, 4)
    blockspergrid = (math.ceil(image.shape[0] / threadsperblock[0]), math.ceil(image.shape[1] / threadsperblock[1]))

    f_kernel[blockspergrid, threadsperblock](dimage, topleft, xstride, ystride, max_iter, *args)
    dimage.to_host()

    dt = timer() - start
    print("Fractal calculated on GPU in %f s" % dt)

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

    dt = timer() - start
    sys.stdout.write('\r' + "Fractal calculated on GPU in %f s" % dt)




