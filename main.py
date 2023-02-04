import cv2
import numpy as np
import math


def main():
    image = cv2.imread("./image.jpeg", 1)
    B, G, R = cv2.split(image)
    Y, U, V = rbg_to_yuv(R, G, B)
    Y, U, V = floor_values(Y, U, V)
    Y, U, V = downsample(Y, U, V)

    # make sure the size of all components are the same
    new_height, new_width = int(Y.shape[0]*2), int(Y.shape[1])*2

    Y = upsample(Y, new_height, new_width, 2)
    U = upsample(U, new_height, new_width, 4)
    V = upsample(V, new_height, new_width, 4)

    Y, U, V = floor_values(Y, U, V)
    R, G, B = (yuv_to_rgb(Y, U, V))

    new_image = cv2.merge([B, G, R]).astype(np.uint8)
    cv2.imshow("Image", new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Convert RGB to YUV
def rbg_to_yuv(r, g, b):
    y = np.zeros(r.shape, dtype=np.float32)
    u = np.zeros(r.shape, dtype=np.float32)
    v = np.zeros(r.shape, dtype=np.float32)

    y = 0.299*r + 0.587*g + 0.114*b
    u = -0.147*r - 0.289*g + 0.436*b + 128
    v = 0.615*r - 0.515*g - 0.100*b + 128

    return y, u, v


# Floor values to get the integer part values will be between 0 and 255
def floor_values(y, u, v):
    Y = np.asarray(np.floor(y), dtype='int')
    U = np.asarray(np.floor(u), dtype='int')
    V = np.asarray(np.floor(v), dtype='int')
    return Y, U, V


# Downsample the image by 2 for Y and 4 for U and V
def downsample(y, u, v):
    Y = y[0::2, 0::2]
    U = u[0::4, 0::4]
    V = v[0::4, 0::4]
    return Y, U, V


# billinear upsampling
def upsample(channel, new_height, new_width, scale_factor):
    img_height, img_width = channel.shape[0], channel.shape[1]
    new_image = np.zeros((new_height, new_width))
    for i in range(new_height):
        for j in range(new_width):
            original_i = i // scale_factor
            original_j = j // scale_factor
            fraction_i = (i % scale_factor) / scale_factor
            fraction_j = (j % scale_factor) / scale_factor
            
            if original_i + 1 >= img_height:
                original_i -= 1
            if original_j + 1 >= img_width:
                original_j -= 1
            
            new_value = (1-fraction_i)*(1-fraction_j) * channel[original_i][original_j] + (1-fraction_i)*fraction_j * channel[original_i][original_j+1] + fraction_i*(
                1-fraction_j) * channel[original_i+1][original_j] + fraction_i*fraction_j * channel[original_i+1][original_j+1]
            new_image[i][j] = new_value
    return new_image


# convert from YUV to RGB
def yuv_to_rgb(y, u, v):
    r = np.zeros(y.shape, dtype=np.float32)
    g = np.zeros(y.shape, dtype=np.float32)
    b = np.zeros(y.shape, dtype=np.float32)

    r = y + 1.4075 * (v - 128)
    g = y - 0.3455 * (u - 128) - (0.7169 * (v - 128))
    b = y + 1.7790 * (u - 128)

    return np.clip(r, 0, 255).astype(np.uint8), np.clip(g, 0, 255).astype(np.uint8), np.clip(b, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    main()
