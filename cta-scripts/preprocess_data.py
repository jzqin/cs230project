"""
Jason Qin
CS230

Preprocess CT scan data by 
1) match segmentation masks to CT scan DICOM images
2) converting segmentation masks and DICOM images to PNG
3) construct relevant dictionary for passing training data into semantic-segmentation-NN
"""

import numpy as np
import matplotlib.image as mpimg
import os
import util
import sys
import argparse

def convert_sample(dcmDir, segFile, outputDir):
    """
    For a given patient, convert DCM images and segmentation masks to PNG files.
    There will be one PNG image per CT scan slice, and one PNG image for each mask
    for each slice.

    Arguments:
        dcmDir: [string] directory containing DCM images for a given patient
        segFile: [string] file containing patient segmentation masks
        outputDir: [string] path to folder to output PNG images

    Return:
        Number of DCM/Nifti images converted
    """
    # make sure outputDir is correctly formatted
    outputDir = os.path.abspath(outputDir)

    # convert patient DCM files to PNG
    dcmFiles = [f for f in os.listdir(dcmDir) if os.isfile(os.join(dcmDir, f))]
    for f in dcmFiles:
        dcmRoot = util.dcm_to_png(f, outputDir)

    # convert patient segmentation mask to PNG
    numSlices = util.nifi_to_png(segFile, outputDir, dcmRoot)

    # if converted a different number of Nifti and DCM images, then there is
    # an issue with the patient data
    if (numSlices != len(dcmFiles)):
        raise RuntimeError:('Different number of Nitfi and DCM images for: {}, {}'.format(dcmDir, segFile))

    return numSlices

def main(args):
    """
    Convert all patient data from DCM and Nifti formats to PNG
    """

    # Create dictionary mapping segmentation masks to CT Scans
    dict = util.seg_map_dict(args.dcm_dict, args.mask_dir))

    # for each patient, convert the DICOM files and masks to png and save these in a new directory
    for key, value in dict.iteritems():
        outputDir = os.path.abspath(value + '/png/')
        convert_sample(value, key, outputDir)

    return None
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dcm_root', '-d', type=str, required=True,
                        help='Root directory containing patient directories. The patient directories contain DCM files for each patient.')
    parser.add_argument('--mask_dir', '-m', type='str', required=True,
                        help='Directory containing segmentation masks.')
    parser.add_argument('--output_file', '-o', type='str', required=True,
                        help='Name of file output file containing training information, formatted to match input to semantic-segmentation-NN.')

    args = parser.parse_args()

    main(args)
