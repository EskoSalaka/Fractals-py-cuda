from numba import complex128
from cmath import isinf, sinh, cosh, exp, phase, log

from colors import *


@cuda.jit('complex128(complex128, float64)', device=True)
def power(z, x):
    return abs(z) ** x * exp(phase(z) * x * complex128(1j))

@cuda.jit('complex128(complex128, complex128)', device=True)
def power(z, c):
    return exp(c * log(z))

@cuda.jit('boolean(complex128, complex128)', device=True)
def is_close(a, b):
    return abs(a-b) <= max(1e-9 * max(abs(a), abs(b)), 1e-9)

@cuda.jit('void(int8[:,:,:], complex128, float64, float64, int32)')
def exp_m(image_array, topleft, xstride, ystride, max_iter):
    y, x = cuda.grid(2)

    if x < image_array.shape[1] and y < image_array.shape[0]:
        c = complex128(topleft + x * xstride - 1j * y * ystride)
        z = c

        i = 0
        while i < max_iter and not isinf(z):
            z = exp(z) + c
            i += 1

        get_log_color_rgb(image_array, x, y, i, max_iter)

@cuda.jit('void(int8[:,:,:], complex128, float64, float64, int32)')
def lambert(image_array, topleft, xstride, ystride, max_iter):
    y, x = cuda.grid(2)

    if x < image_array.shape[1] and y < image_array.shape[0]:
        c = complex128(topleft + x * xstride - complex128(1j) * y * ystride)

        c = exp(c * exp(-c))
        z = c
        o = complex128(0.0)

        for i in range(max_iter):
            z = power(c, z)

            if isinf(z):
                get_log_color_rgb(image_array, x, y, i, max_iter)
                return

            if is_close(z, o):
                get_log_color_b(image_array, x, y, i, max_iter)
                return

            if i % 3 == 0:
                o = z




@cuda.jit('void(int8[:,:,:], complex128, float64, float64, int32)')
def mandelbrot(image_array, topleft, xstride, ystride, max_iter):
    y, x = cuda.grid(2)

    if x < image_array.shape[1] and y < image_array.shape[0]:
        c = complex128(topleft + x * xstride - 1j * y * ystride)
        z = 0

        i = 0
        while i < max_iter and z.real * z.real + z.imag * z.imag < 4:
            z = z * z + c
            i += 1

        get_log_color_rgb(image_array, x, y, i, max_iter)



@cuda.jit('void(int8[:,:,:], complex128, float64, float64, int32, int32, int32)')
def mandelbrot_split(image_array, topleft, xstride, ystride, max_iter, split_start, split_end):
    y, x = cuda.grid(2)
    y = y + split_start

    if x < image_array.shape[1] and y < split_end:
        c = complex128(topleft + x * xstride - 1j * y * ystride)
        z = 0

        i = 0
        while i < max_iter and z.real * z.real + z.imag * z.imag < 4:
            z = z * z + c
            i += 1

        get_log_color_rgb(image_array, x, y, i, max_iter)