import numpy
import random


def create_red_noise( width, height, r=10):
    """
    Create red noise RGB image

    Parameters
    ----------
    outfile : str
    width : int
    height : int
    r : int
        Random maximum offset compared to the last pixel
    """
    array = numpy.random.rand(height, width) * 255
    for x in range(width):
        for y in range(height):
            if y == 0:
                if x == 0:
                    continue
                else:
                        array[y][x] = (array[y][x-1] +
                                          random.randint(-r, r))
            else:
                if x == 0:
                        array[y][x] = (array[y-1][x] +
                                          random.randint(-r, r))
                else:
                        array[y][x] = (((array[y-1][x] +
                                            array[y][x-1]) / 2.0 +
                                           random.randint(-r, r)))
    return array
    
    
def normalize(array):
    return array/numpy.linalg.norm(array)
    
    
def norm(array):
    return numpy.linalg.norm(array)