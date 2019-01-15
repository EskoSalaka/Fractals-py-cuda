from base import *
from kernels import *

if __name__ == "__main__":
    Explorer(sinhcosh, -5, 5, -5, 5, 1000, 1000, interpolation='bilinear')

