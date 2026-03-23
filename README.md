# Mathematical Methods of Image Processing — Labs

## Table of contents

- [Mathematical Methods of Image Processing — Labs](#mathematical-methods-of-image-processing--labs)
  - [Table of contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Lab 1](#lab-1)
  - [Lab 2](#lab-2)
  - [Lab 3](#lab-3)
  - [Lab 4](#lab-4)
  - [Lab 5](#lab-5)

## Introduction

Even though the course is about *image processing*, the laboratory part is primarily **signal-processing tools** (sampling/quantization, filtering/convolution, Fourier analysis). These 1D/2D tools are the foundation for many image-processing tasks.

How to work with labs in this repo:
- Implement the required functions by following the **docstrings + function interfaces** in each `labs/labXX_*.py`.
     Run the lab file as a script to generate outputs (no GUI required): `python labs/lab01_filtering_convolution_fft.py`

Install deps: `pip install -r requirements.txt`

## Lab 1

**Curriculum topic:** Topic 1 - Python tools for signal processing.  
**What you get:** practical spatial filtering + FFT utilities on real images (noise removal, edge detection, frequency-domain filtering).

Implementation module:
- `labs/lab01_filtering_convolution_fft.py`

Implemented functionality (high level):
- Spatial filtering: `conv2d`, box blur, Gaussian blur, median blur
- Noise synthesis: salt & pepper, Gaussian noise (seeded)
- Edges: Sobel (gx/gy/magnitude), Laplacian
- FFT utilities (OpenCV DFT pattern): spectrum, shift, magnitude spectrum, ideal LP/HP masks, apply frequency filter

## Lab 2

**Curriculum topic:** Topic 3 - Wavelets and multi-scale analysis (+ STFT bridge).  
**What you get:** Haar wavelet tools for 1D/2D signals, multi-level denoising by coefficient thresholding, and STFT/spectrogram utilities for 1D time-frequency analysis.

Implementation module:
- `labs/lab02_wavelets_stft.py`

Implemented functionality (high level):
- 1D Haar transform and inverse: `haar_dwt1`, `haar_idwt1` (with documented odd-length padding behavior)
- 2D separable Haar transform and inverse: `haar_dwt2`, `haar_idwt2` with LL/LH/HL/HH bands
- Wavelet coefficient thresholding: `wavelet_threshold` (hard/soft, arrays and nested tuples/lists)
- Multi-level denoising: `wavelet_denoise` (deterministic and shape-preserving)
- STFT bridge utilities: `stft1` and `spectrogram_magnitude`

## Lab 3

**Curriculum topic:** Topic 3 - Geometric transformations + feature detection/matching.  
**What you get:** affine/perspective warps, ORB keypoint detection, descriptor matching, and RANSAC homography estimation.

Implementation module:
- `labs/lab03_geometry_features_matching.py`

Implemented functionality (high level):
- Geometric warping: `warp_affine`, `warp_perspective`
- ORB feature extraction: `detect_orb`
- Descriptor matching with ratio test: `match_descriptors`
- Homography estimation with RANSAC: `estimate_homography_from_matches`

## Lab 4

**Curriculum topic:** Topic 4 - Markov Random Fields for image restoration.  
**What you get:** pairwise MRF denoising with data + smoothness energy, supporting quadratic and Huber penalties.

Implementation module:
- `labs/lab04_mrf_restoration.py`

Implemented functionality (high level):
- MRF energy computation: `mrf_energy`
- Iterative MRF denoising (gradient descent): `mrf_denoise`
- Smoothness penalties: quadratic and Huber

## Lab 5

**Curriculum topic:** Topic 7 - Motion estimation.  
**What you get:** dense optical flow with Farneback and HSV/BGR flow visualization.

Implementation module:
- `labs/lab05_motion_estimation.py`

Implemented functionality (high level):
- Dense optical flow estimation (Farneback): `optical_flow_farneback`
- Flow visualization in BGR via HSV mapping: `flow_to_hsv`
