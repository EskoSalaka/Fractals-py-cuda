import math
import numba as nb

from numba import cuda
from cmath import isinf, sinh, cosh, exp, phase


@cuda.jit('void(int8[:,:,:], int32, int32, int32, int32)', device=True)
def get_color1(image_array, x, y, iterations, max_iterations):
    if iterations == max_iterations:
        return

    k = 3.0 * math.log(nb.float64(iterations)) / math.log(nb.float64(max_iterations) - 1.0)

    if k < 1:
        image_array[y, x, 0] = nb.int8(255 * k)
        image_array[y, x, 1] = 0
        image_array[y, x, 2] = 0
    elif k < 2:
        image_array[y, x, 0] = 255
        image_array[y, x, 1] = nb.int8(255 * (k - 1))
        image_array[y, x, 2] = 0
    else:
        image_array[y, x, 0] = 255
        image_array[y, x, 1] = 255
        image_array[y, x, 2] = nb.int8(255 * (k - 2))

@cuda.jit('void(int8[:,:,:], int32, int32, int32, int32)', device=True)
def get_color2(image_array, x, y, iterations, max_iterations):
    if iterations == max_iterations:
        return

    k = 3.0 * math.log(nb.float64(iterations)) / math.log(nb.float64(max_iterations) - 1.0)

    if k < 1:
        image_array[y, x, 0] = 0
        image_array[y, x, 1] = 0
        image_array[y, x, 2] = nb.int8(255 * k)
    elif k < 2:
        image_array[y, x, 0] = 0
        image_array[y, x, 1] = nb.int8(255 * (k - 1))
        image_array[y, x, 2] = 255
    else:
        image_array[y, x, 0] = nb.int8(255 * (k - 2))
        image_array[y, x, 1] = 255
        image_array[y, x, 2] = 255


@cuda.jit('complex128(complex128, float64)', device=True)
def power(c, x):
    return abs(c) ** x * exp(phase(c) * x * 1j)

@cuda.jit('boolean(complex128, complex128)', device=True)
def is_close(a, b):
    return abs(a-b) <= max(1e-9 * max(abs(a), abs(b)), 0)

@cuda.jit('void(int8[:,:,:], complex128, float64, float64, int32)')
def exp_m(image, topleft, xstride, ystride, max_iter):
    y, x = cuda.grid(2)

    if x < image.shape[1] and y < image.shape[0]:
        c = nb.complex128(topleft + x * xstride - 1j * y * ystride)
        z = c

        i = 0
        while i < max_iter and not isinf(z):
            z = exp(z) + c
            i += 1

        get_color1(image, x, y, i, max_iter)

@cuda.jit('void(int8[:,:,:], complex128, float64, float64, int32)')
def mandelbrot(image, topleft, xstride, ystride, max_iter):
    y, x = cuda.grid(2)

    if x < image.shape[1] and y < image.shape[0]:
        c = nb.complex128(topleft + x * xstride - 1j * y * ystride)
        z = c

        i = 0
        while i < max_iter and abs(z) < 4:
            z =z*z + c
            i += 1

        get_color1(image, x, y, i, max_iter)



@cuda.jit('void(int8[:,:,:], complex128, float64, float64, int32, int32, int32)')
def mandelbrot_split(image, topleft, xstride, ystride, max_iter, split_start, split_end):
    y, x = cuda.grid(2)
    y = y + split_start

    if x < image.shape[1] and y < split_end:
        c = complex(topleft.real + x * xstride, topleft.imag - y * ystride)
        z = c

        i = 0
        while i < max_iter and abs(z) < 4:
            z = z*z + c
            i += 1

        get_color1(image, x, y, i, max_iter)