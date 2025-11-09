# Calculates the output shape given the imput shape and parameters
import math

def get_out_shape(x, y, z, padding, dialation, size, stride):

    x_out = math.floor(((x + 2 * padding - dialation * (size-1) - 1)/stride) + 1)
    y_out = math.floor(((y + 2 * padding - dialation * (size-1) - 1)/stride) + 1)
    z_out = math.floor(((z + 2 * padding - dialation * (size-1) - 1)/stride) + 1)

    return (x_out, y_out, z_out)

print(get_out_shape(11, 14, 11, 0, 1, 3, 2))