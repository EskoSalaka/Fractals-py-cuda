from base import Explorer
from kernels import mandelbrot, mandelbrot_split, lambert

if __name__ == "__main__":
    Explorer(lambert, -5, 2, -2, 2, 1000, 1000, interpolation='bilinear').show()


