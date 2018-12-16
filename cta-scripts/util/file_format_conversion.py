"""
Jason Qin
CS230

Utility functions for converting medical image types (currently: DICOM and NRRD) to PNG, which 
can be fed into semantic segmentation NN.
"""

import os
import nrrd
import numpy as np
import pydicom as dicom
import matplotlib.image as mpimg
import imageio as imgio
import re

def dcm_to_png(dcmFile, outputDir):
    """
    Convert DICOM images to PNG
    Arguments:
        dcmFile: [string] path to dcmFile to convert
        outputDir: [string] path to folder to output PNG image

    Return:
        Root name of DICOM file converted (IM-####-)
    """
    with open(dcmFile, 'rb') as d:
        dcmData = dicom.dcmread(d)
    dcmData = dcmData.pixel_array

    dcmName = re.search(r'IM-(\d+)-(\d+).dcm', os.path.basename(dcmFile))
    if dcmName is None:
        raise RuntimeError('Unexpected DICOM name: {}'.format(os.path.basename(dcmFile)))

    outputDir = os.path.abspath(outputDir) + '/'
    dcmName = dcmName.group(0)
    imgio.imwrite(outputDir + dcmName[:-4] + '.png', dcmData)

    # return the root name of the file split
    dcmRoot = dcmName.split('-')
    dcmRoot = '{}-{}-'.format(dcmRoot[0], dcmRoot[1])
    return dcmRoot


def nrrd_to_png(nrrdFile, outputDir, rootName):
    """
    Convert NRRD to PNG
    Arguments:
        nrrdFile: [string] path to nrrdFile to convert
        outputDir: [string] path to folder to output PNG image
        rootName: [string] root of patient data names, should be of format IM-####-

    Return: 
        Number of PNGs created from segmentation NRRD
    """
    mask = nrrd.read(nrrdFile)
    mask = mask[0].astype(np.uint8)
    
    # NRRD data is stored as (X, Y, Z) numpy arrays, with X = height,
    # Y = width, and Z = # slices of CT scan data
    numSlices = mask.shape[2]

    outputDir = os.path.abspath(outputDir) + '/'
    for i in range(numSlices):
        maskData = mask[:, :, i].transpose(1, 0)  # must transpose for mask to match DCM data!!!
        maskName = rootName + '{0:04}_mask.png'.format(i+1)
        imgio.imsave(outputDir + maskName, maskData)

    return numSlices
