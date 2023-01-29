import cv2
import numpy as np


def main():
    image = cv2.imread("./image.jpeg", 1)
    B, G, R = cv2.split(image)
    Y, U, V = floor_values(rbg_to_yuv(R, G, B))

    
    print(Y)

def rbg_to_yuv(r, g, b):
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.1471 * r - 0.2889 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g - 0.1 * b
    return y, u, v

def floor_values(y, u, v):
    Y = np.asarray(np.floor(y), dtype='int')
    U = np.asarray(np.floor(u), dtype='int')
    V = np.asarray(np.floor(v), dtype='int')
    return Y, U, V


def downsample(y, u, v):
    pass
    # return Y, U, V


if __name__ == "__main__":
    main()
