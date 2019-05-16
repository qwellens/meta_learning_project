import os
import numpy as np
from numpy.fft import fftshift, ifftshift, fft2 as nfft2, ifft2 as nifft2
import matplotlib.pyplot as plt
import PIL
import glob

def png_to_array(img, zero_loc="center"):
    """
    Load a png image as a numpy array.
    """
    im_arr = np.frombuffer(img.convert('L').tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((img.size[1], img.size[0])) # keep just rows and cols (omniglot is black and white)
    if zero_loc == 'center':
        im_arr = ifftshift(im_arr)
    return im_arr.astype(np.float64)/255

def array_to_png(array, fname, zero_loc="center", normalize=False, zoom=1):
    """
    Write and save a numpy array as a png image.
    """
    assert zero_loc in {'center', 'topleft'}, "zero_loc must be 'center' or 'topleft', not %r" % zero_loc
    if zero_loc == 'center': # center each image
        array = fftshift(array)
    if normalize:
        array = array - np.min(array)
        array = array / np.max(array)
    h, w = array.shape
    out = PIL.Image.new(mode='L', size=(w, h))
    out.putdata(np.around(255*array).reshape(w*h))
    out.resize((int(w*zoom), int(h*zoom)), PIL.Image.NEAREST).save(fname)

def show_image(array, colorscale='linear', zero_loc='center', cmap='gray',
               show=True, title='', xlabel='', ylabel='', vmax=None, vmin=None):
    """
    Display an image using matplotlib - useful for visualizing processing results.
    """
    assert zero_loc in {'center', 'topleft'}, "zero_loc must be 'center' or 'topleft', not %r" % zero_loc
    array = array.real
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if colorscale == 'log':
        array[array != 0] = np.log10(array[array != 0])
        array[array == 0] = np.min(array-1)
    h, w = array.shape
    if zero_loc == 'center':
        bounds = [-w//2-0.5+(w%2), w//2-0.5+(w%2), h//2-0.5+(w%2), -h//2-0.5+(w%2)]
        array = fftshift(array)
    else:
        bounds = [0, w, h, 0]
    i = plt.imshow(array, cmap=cmap, interpolation='nearest', origin='upper',
                   extent=bounds, vmin=vmin, vmax=vmax)
    plt.colorbar()
    if show:
        plt.show()

def pepper_and_salt(img, svp=0.5, level=0.007):
    """
    Adds pepper and salt noise (each pixel value becomes 0, or 1).
    """
    row,col = img.shape
    noisy_img = np.copy(img)

    # Salt
    num_salt = np.ceil(level * img.size * svp)
    coords = [np.random.randint(0, i - 1, int(num_salt))
        for i in img.shape]
    noisy_img[coords] = 1

    # Pepper
    num_pepper = np.ceil(level* img.size * (1. - svp))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
        for i in img.shape]
    noisy_img[coords] = 0
    return noisy_img

def gaussian_blur(img, mean=0, std=0.1):
    """
    Adds gaussian blur to an image.
    """
    row, col = img.shape
    gauss = np.random.normal(mean, std, (row,col))
    gauss = gauss.reshape(row,col)
    noisy_img = img + gauss
    return noisy_img

def speckle(img):
    """
    Adds speckle noise to an image.
    """
    row,col = img.shape
    gauss = np.random.randn(row,col)
    gauss = gauss.reshape(row,col)        
    noisy_img = img + (img * gauss)
    return noisy_img

def poisson(img):
    """
    Adds poisson random noise to an image.
    """
    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy_img = np.random.poisson(img * vals) / float(vals)
    return noisy_img

if __name__ == '__main__':

    path = '*/*/'
    images = glob.glob(path + '*')

    noise_mode = 'pepper_and_salt' # gaussian, pepper_and_salt, speckle, poisson

    for img in images:
        fname = img
        print(fname)
        img = PIL.Image.open(img)
        img_resized = img.resize((28, 28), resample=PIL.Image.LANCZOS) # resize image: done the same in original paper
        img_array = png_to_array(img)

        if noise_mode == 'pepper_and_salt': # replace random values with 0 or 1
            img_array = pepper_and_salt(img_array)
        elif noise_mode == 'gaussian': # gaussian noise with mean/std defined
            img_array = gaussian_blur(img_array)
        elif noise_mode == 'speckle': # multiplicative noise from a distribution
            img_array = speckle(img_array)
        elif noise_mode == 'poisson': # poisson noise sampled from the images distribution
            img_array = poisson(img_array)

        array_to_png(img_array, fname)
    








