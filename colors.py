from numba import cuda
from math import log as real_log
from numba import float64, int8


@cuda.jit('void(int8[:,:,:], int32, int32, int32, int32)', device=True)
def get_log_color_rgb(image_array, x, y, iterations, max_iterations):
    if iterations == max_iterations:
        return

    k = 3.0 * real_log(float64(iterations)) / real_log(float64(max_iterations))

    if k < 1:
        image_array[y, x, 0] = int8(255 * k)
        image_array[y, x, 1] = 0
        image_array[y, x, 2] = 0
    elif k < 2:
        image_array[y, x, 0] = 255
        image_array[y, x, 1] = int8(255 * (k - 1))
        image_array[y, x, 2] = 0
    else:
        image_array[y, x, 0] = 255
        image_array[y, x, 1] = 255
        image_array[y, x, 2] = int8(255 * (k - 2))


@cuda.jit('void(int8[:,:,:], int32, int32, int32, int32)', device=True)
def get_log_color_bgr(image_array, x, y, iterations, max_iterations):
    if iterations == max_iterations:
        return

    k = 3.0 * real_log(float64(iterations)) / real_log(float64(max_iterations) - 1.0)

    if k < 1:
        image_array[y, x, 0] = 0
        image_array[y, x, 1] = 0
        image_array[y, x, 2] = int8(255 * k)
    elif k < 2:
        image_array[y, x, 0] = 0
        image_array[y, x, 1] = int8(255 * (k - 1))
        image_array[y, x, 2] = 255
    else:
        image_array[y, x, 0] = int8(255 * (k - 2))
        image_array[y, x, 1] = 255
        image_array[y, x, 2] = 255

@cuda.jit('void(int8[:,:,:], int32, int32, int32, int32)', device=True)
def get_log_color_gray(image_array, x, y, iterations, max_iterations):
    if iterations == max_iterations:
        return

    k = real_log(float64(iterations)) / real_log(float64(max_iterations) - 1.0)

    image_array[y, x, 0] = int8(255 * k)
    image_array[y, x, 1] = int8(255 * k)
    image_array[y, x, 2] = int8(255 * k)

@cuda.jit('void(int8[:,:,:], int32, int32, int32, int32)', device=True)
def get_log_color_r(image_array, x, y, iterations, max_iterations):
    if iterations == max_iterations:
        return

    k = real_log(float64(iterations)) / real_log(float64(max_iterations) - 1.0)

    image_array[y, x, 0] = int8(255 * k)

@cuda.jit('void(int8[:,:,:], int32, int32, int32, int32)', device=True)
def get_log_color_g(image_array, x, y, iterations, max_iterations):
    if iterations == max_iterations:
        return

    k = real_log(float64(iterations)) / real_log(float64(max_iterations) - 1.0)

    image_array[y, x, 1] = int8(255 * k)

@cuda.jit('void(int8[:,:,:], int32, int32, int32, int32)', device=True)
def get_log_color_b(image_array, x, y, iterations, max_iterations):
    if iterations == max_iterations:
        return

    k = real_log(float64(iterations)) / real_log(float64(max_iterations) - 1.0)

    image_array[y, x, 2] = int8(255 * k)