#!/usr/bin/python
#
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Python implementation of MS-SSIM.

Usage:

python msssim.py --original_image=original.png --compared_image=distorted.png
"""
import numpy as np
from scipy import signal
from scipy.ndimage.filters import convolve
import tensorflow as tf
import random
from tqdm import trange
tf.flags.DEFINE_string('original_image', None, 'Path to PNG image.')
tf.flags.DEFINE_string('compared_image', None, 'Path to PNG image.')
tf.flags.DEFINE_string('path', None, 'directory path to iamges.')
tf.flags.DEFINE_integer('nsamples', 0, 'Number of samples.')
FLAGS = tf.flags.FLAGS
def _FSpecialGauss(size, sigma):
  """Function to mimic the 'fspecial' gaussian MATLAB function."""
  radius = size // 2
  offset = 0.0
  start, stop = -radius, radius + 1
  if size % 2 == 0:
    offset = 0.5
    stop -= 1
  x, y = np.mgrid[offset + start:stop, offset + start:stop]
  assert len(x) == size
  g = np.exp(-((x**2 + y**2)/(2.0 * sigma**2)))
  return g / g.sum()


def _SSIMForMultiScale(img1, img2, max_val=255, filter_size=11,
                       filter_sigma=1.5, k1=0.01, k2=0.03):
  """Return the Structural Similarity Map between `img1` and `img2`.
  This function attempts to match the functionality of ssim_index_new.m by
  Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
  Arguments:
    img1: Numpy array holding the first RGB image batch.
    img2: Numpy array holding the second RGB image batch.
    max_val: the dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Size of blur kernel to use (will be reduced for small images).
    filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
      for small images).
    k1: Constant used to maintain stability in the SSIM calculation (0.01 in
      the original paper).
    k2: Constant used to maintain stability in the SSIM calculation (0.03 in
      the original paper).
  Returns:
    Pair containing the mean SSIM and contrast sensitivity between `img1` and
    `img2`.
  Raises:
    RuntimeError: If input images don't have the same shape or don't have four
      dimensions: [batch_size, height, width, depth].
  """
  if img1.shape != img2.shape:
    raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                       img1.shape, img2.shape)
  if img1.ndim != 4:
    raise RuntimeError('Input images must have four dimensions, not %d',
                       img1.ndim)

  img1 = img1.astype(np.float64)
  img2 = img2.astype(np.float64)
  _, height, width, _ = img1.shape

  # Filter size can't be larger than height or width of images.
  size = min(filter_size, height, width)

  # Scale down sigma if a smaller filter size is used.
  sigma = size * filter_sigma / filter_size if filter_size else 0

  if filter_size:
    window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
    sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
    sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
  else:
    # Empty blur kernel so no need to convolve.
    mu1, mu2 = img1, img2
    sigma11 = img1 * img1
    sigma22 = img2 * img2
    sigma12 = img1 * img2

  mu11 = mu1 * mu1
  mu22 = mu2 * mu2
  mu12 = mu1 * mu2
  sigma11 -= mu11
  sigma22 -= mu22
  sigma12 -= mu12

  # Calculate intermediate values used by both ssim and cs_map.
  c1 = (k1 * max_val) ** 2
  c2 = (k2 * max_val) ** 2
  c4 = .1 ** 8
  #v1 = 2.0 * sigma12 + c2
  v1 = 1.0 * sigma12 + np.sqrt(sigma11+c4) * np.sqrt(sigma22+c4) + c2
  v2 = sigma11 + sigma22 + c2
  ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
  cs = np.mean(v1 / v2)
  return ssim, cs


def MultiScaleSSIM(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5,
                   k1=0.01, k2=0.03, weights=None):
  """Return the MS-SSIM score between `img1` and `img2`.
  This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
  Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
  similarity for image quality assessment" (2003).
  Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf
  Author's MATLAB implementation:
  http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
  Arguments:
    img1: Numpy array holding the first RGB image batch.
    img2: Numpy array holding the second RGB image batch.
    max_val: the dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Size of blur kernel to use (will be reduced for small images).
    filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
      for small images).
    k1: Constant used to maintain stability in the SSIM calculation (0.01 in
      the original paper).
    k2: Constant used to maintain stability in the SSIM calculation (0.03 in
      the original paper).
    weights: List of weights for each level; if none, use five levels and the
      weights from the original paper.
  Returns:
    MS-SSIM score between `img1` and `img2`.
  Raises:
    RuntimeError: If input images don't have the same shape or don't have four
      dimensions: [batch_size, height, width, depth].
  """
  if img1.shape != img2.shape:
    raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                       img1.shape, img2.shape)
  if img1.ndim != 4:
    raise RuntimeError('Input images must have four dimensions, not %d',
                       img1.ndim)

  # Note: default weights don't sum to 1.0 but do match the paper / matlab code.
  weights = np.array(weights if weights else
                     [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
  levels = weights.size
  downsample_filter = np.ones((1, 2, 2, 1)) / 4.0
  im1, im2 = [x.astype(np.float64) for x in [img1, img2]]
  mssim = np.array([])
  mcs = np.array([])
  for _ in range(levels):
    ssim, cs = _SSIMForMultiScale(
        im1, im2, max_val=max_val, filter_size=filter_size,
        filter_sigma=filter_sigma, k1=k1, k2=k2)
    mssim = np.append(mssim, ssim)
    mcs = np.append(mcs, cs)
    filtered = [convolve(im, downsample_filter, mode='reflect')
                for im in [im1, im2]]
    im1, im2 = [x[:, ::2, ::2, :] for x in filtered]
  return (np.prod(mcs[0:levels-1] ** weights[0:levels-1]) *
          (mssim[levels-1] ** weights[levels-1]))

def pairwise_distance(path, nsamples=100):
    if path is None:
        print('Please enter image path.')
        return
    from os.path import isfile, join
    from os import listdir
    import cv2
    import matplotlib.pyplot as plt
    from tqdm import trange
    def listfile(path):
        return [join(path,f) for f in listdir(path) if isfile(join(path, f)) and ('.jpeg' in f or '.jpg' in f or '.png' in f)]
    def read_img(path, npx = 64):
        img = plt.imread(path)
        if 'celeba' in path.lower() and 'data' in path.lower():
            img = img[50:50+128,25:25+128,:] 
        return cv2.resize(img, dsize=(npx, npx), interpolation=cv2.INTER_AREA).astype('int32').reshape(-1,npx,npx,3)
    files = listfile(path)
    #sampled_files = [ files[i] for i in np.random.randint(low=0, high=len(files), size=nsamples)]
    #sampled_files_i = [ files[i] for i in np.random.randint(low=0, high=len(files), size=nsamples)]
    #sampled_files_j = [ files[i] for i in np.random.randint(low=0, high=len(files), size=nsamples)]
    sampled_files = [ files[i] for i in np.random.permutation(len(files))[:nsamples*2]]
    sampled_files_i = sampled_files[::2]
    sampled_files_j = sampled_files[1::2]

    #print(sampled_files_i)
    #print(sampled_files_j)
    img = read_img(sampled_files_i[0])
    print('Image shape: %s'%(img.shape,))
    msssim_scores = np.zeros(nsamples)
    for i in trange(nsamples):
        msssim_scores[i]=MultiScaleSSIM(read_img(sampled_files_i[i]), read_img(sampled_files_j[i]), max_val=255)
    #print(np.mean(msssim_scores))
    return(np.mean(msssim_scores))
def pairwise_distance_imgs(imgs, nsamples=100):
    np.random.shuffle(imgs)
    imgs = imgs[:nsamples*2]
    msssim_scores = np.zeros(nsamples)
    for i in trange(nsamples):
        msssim_scores[i]=MultiScaleSSIM(imgs[i:i+1], imgs[i+nsamples:i+nsamples+1], max_val=255)
    #print(np.mean(msssim_scores))
    return(np.mean(msssim_scores))
    
def variance(path, nsamples=100):
    if path is None:
        print('Please enter image path.')
        return
    from os.path import isfile, join
    from os import listdir
    import cv2
    import matplotlib.pyplot as plt

    def listfile(path):
        return [join(path,f) for f in listdir(path) if isfile(join(path, f)) and ('.jpeg' in f or '.jpg' in f or '.png' in f)]
    def read_img(path, npx = 64):
        img = plt.imread(path)
        if 'celeba' in path.lower() and 'data' in path.lower():
            img = img[50:50+128,25:25+128,:] 
        return cv2.resize(img, dsize=(npx, npx), interpolation=cv2.INTER_AREA).astype('int32').reshape(-1,npx,npx,3)
    files = listfile(path)
    #sampled_files = [ files[i] for i in np.random.randint(low=0, high=len(files), size=nsamples)]
    sampled_files = [ files[i] for i in np.random.permutation(len(files))[:nsamples]]
    img_list = []
    #print('Reading images from disk.')
    for i in trange(len(sampled_files)):
        img_list.append(read_img(sampled_files[i]))
    imgs = np.concatenate(img_list)
    mean_img = np.mean(imgs, axis=0).reshape(-1,64,64,3)
    tmp = []
    #print('Computing variance.')
    for i in trange(nsamples):
        tmp.append(MultiScaleSSIM(imgs[i:i+1],mean_img))
    #print(np.sum(tmp)/(nsamples-1))
    return np.sum(tmp)/(nsamples)

def variance_imgs(imgs, nsamples=100):
    np.random.shuffle(imgs)
    imgs = imgs[:nsamples]
    mean_img = np.mean(imgs, axis=0).reshape(-1,64,64,3)
    tmp = []
    #print('Computing variance.')
    for i in trange(nsamples):
        tmp.append(MultiScaleSSIM(imgs[i:i+1],mean_img))
    #print(np.sum(tmp)/(nsamples-1))
    return np.sum(tmp)/(nsamples)
def difference_imgs(imgs, ref, nsamples=100):
    np.random.shuffle(imgs)
    imgs = imgs[:nsamples]
    tmp = []
    for i in trange(nsamples):
        tmp.append(MultiScaleSSIM(imgs[i:i+1], ref))
    return np.mean(tmp)
def main(_):
  print('main')
  if FLAGS.original_image is None or FLAGS.compared_image is None:
    print('\nUsage: python msssim.py --original_image=original.png '
          '--compared_image=distorted.png\n\n')
    return

  if not tf.gfile.Exists(FLAGS.original_image):
    print('\nCannot find --original_image.\n')
    return

  if not tf.gfile.Exists(FLAGS.compared_image):
    print('\nCannot find --compared_image.\n')
    return

  with tf.gfile.FastGFile(FLAGS.original_image, 'rb') as image_file:
    img1_str = image_file.read()
  with tf.gfile.FastGFile(FLAGS.compared_image, 'rb') as image_file:
    img2_str = image_file.read()

  input_img = tf.placeholder(tf.string)
  decoded_image = tf.expand_dims(tf.image.decode_image(input_img, channels=3), 0)

  with tf.Session() as sess:
    img1 = sess.run(decoded_image, feed_dict={input_img: img1_str})
    img2 = sess.run(decoded_image, feed_dict={input_img: img2_str})
  import numpy as np
  print(np.max(img1), np.min(img2))
  print((MultiScaleSSIM(img1, img2, max_val=255)))


#if __name__ == '__main__':
#  tf.app.run(main=pairwise_distance)
