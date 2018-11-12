"""
Jason Qin
CS230

Utility functions for converting medical image types (currently: DICOM and Nifti) to PNG, which 
can be fed into semantic segmentation NN.
"""

import nibabel as nib
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

    dcmName = re.search(r'(\d+)-(\d+).dcm', os.path.basename(dcmFile))
    if dcmName is None:
        raise RuntimeError:('Unexpected DICOM name: {}'.format(os.path.basename(dcmFile)))

    outputDir = os.path.abspath(outputDir)
    plt.imsave(outputDir + dcmName[:-4] + '.png', dcmData)

    # return the root name of the file split
    dcmRoot = dcmName.split('-')
    dcmRoot = '{}-{}-'.format(dcmRoot[0], dcmRoot[1])
    return dcmRoot


def nifti_to_png(niftiFile, outputDir, rootName):
    """
    Convert Nifti to PNG
    Arguments:
        niftiFile: [string] path to niftiFile to convert
        outputDir: [string] path to folder to output PNG image
        rootName: [string] root of patient data names, should be of format IM-####-

    Return: 
        Number of PNGs created from segmentation Nifti
    """
    mask = nib.load(niftiFile)

    # Nifti data is stored as (X, Y, Z) numpy arrays, with X = height,
    # Y = width, and Z = # slices of CT scan data
    numSlices = mask.shape[2]

    outputDir = os.path.abspath(outputDir)
    for i in range(numSlices):
        maskData = mask[:, :, i].transpose(1, 0)  # must transpose for mask to match DCM data!!!
        maskName = rootName + '{0:04}_mask.png'.format(i+1)
        imgio.imsave(outputDir + maskName, maskData)

    return numSlices
