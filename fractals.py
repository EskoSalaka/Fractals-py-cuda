from base import *
from kernels import *

if __name__ == "__main__":
    Explorer(mandelbrot, -2, 1, -1, 1, 1000, 1000, interpolation='bilinear')

