# Mathematical Methods of Image Processing — Labs

## Table of contents

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

TBD

## Lab 3

TBD

## Lab 4

TBD

## Lab 5

TBD
