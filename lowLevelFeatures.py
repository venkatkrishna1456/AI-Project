
import cv2
import numpy as np
from skimage.feature import hog
from skimage import data, exposure
#import mahotas as mt

# function to compute RGB histogram
def rgbHist(image, num_bins = 255):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r_hist, _ = np.histogram(image[:, :, 0], bins=num_bins, range=(0, 256))
    r_hist = r_hist / np.linalg.norm(r_hist)
    g_hist, _ = np.histogram(image[:, :, 1], bins=num_bins, range=(0, 256))
    g_hist = g_hist / np.linalg.norm(g_hist)
    b_hist, _ = np.histogram(image[:, :, 2], bins=num_bins, range=(0, 256))
    b_hist = b_hist / np.linalg.norm(b_hist)
    return np.concatenate((r_hist, g_hist, b_hist), axis=None)

# function to compute HSV histogram
def hsvHist(image, num_bins = 255):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_hist, _ = np.histogram(image[:, :, 0], bins=num_bins, range=(0, 256))
    h_hist = h_hist / np.linalg.norm(h_hist)
    s_hist, _ = np.histogram(image[:, :, 1], bins=num_bins, range=(0, 256))
    s_hist = s_hist / np.linalg.norm(s_hist)
    v_hist, _ = np.histogram(image[:, :, 2], bins=num_bins, range=(0, 256))
    v_hist = v_hist / np.linalg.norm(v_hist)
    return np.concatenate((h_hist, s_hist, v_hist), axis=None)

# function to compute HoG histogram
def HoG(image, num_bins, rescaled = True, max_val = 10, orientations = 8):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fd, hog_image = hog(image, orientations = orientations, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)
    if rescaled:
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, max_val))
        hog_feature, _ = np.histogram(hog_image_rescaled, bins=num_bins, range=(0, max_val))
        hog_feature / np.linalg.norm(hog_feature)
        hog_feature = hog_feature / np.linalg.norm(hog_feature)
        return hog_feature
    else:
        hog_feature, _ = np.histogram(hog_image, bins=num_bins, range=(0, 132))
        hog_feature = hog_feature / np.linalg.norm(hog_feature)
        return hog_feature

# function to compute Haralick textural features
#def haralickFeat(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture_feature = mt.features.haralick(image)
    haralick_feature = texture_feature.mean(axis = 0)
    haralick_feature = haralick_feature / np.linalg.norm(haralick_feature)
    return haralick_feature

