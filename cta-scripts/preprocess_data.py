"""
Jason Qin
CS230

Preprocess CT scan data by 
1) match segmentation masks to CT scan DICOM images
2) converting segmentation masks and DICOM images to PNG
3) construct relevant dictionary for passing training data into semantic-segmentation-NN
"""

import os
import sys
import json
import argparse
import numpy as np
import imageio as imgio
import util


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
        Number of DCM/NRRD images converted
    """
    # make sure directory names are correctly formatted
    dcmDir = os.path.abspath(dcmDir) + '/'
    outputDir = os.path.abspath(outputDir) + '/'

    # make sure output directory exists
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)
    
    # convert patient DCM files to PNG
    dcmFiles = [f for f in os.listdir(dcmDir) if os.path.isfile(os.path.join(dcmDir, f))]
    # import pdb; pdb.set_trace()
    for f in dcmFiles:
        dcmRoot = util.dcm_to_png(dcmDir + f, outputDir)

    # convert patient segmentation mask to PNG
    numSlices = util.nrrd_to_png(segFile, outputDir, dcmRoot)

    # if converted a different number of NRRD and DCM images, then there is
    # an issue with the patient data
    if (numSlices != len(dcmFiles)):
        raise RuntimeError('Different number of Nitfi and DCM images for: {}, {}'.format(dcmDir, segFile))

    return numSlices

def build_NN_inputs(pngDir, jsonFile):
    """
    For a directory containing PNG images of CT scans and segmentation masks, create
    relevant JSON data to load each pair of CT scan/mask images into the semantic-segmentation-NN

    Arguments:
        pngDir: [string] directory containing PNG images of CT scans and segmentation masks
        jsonFile: [string] jsonFile containing data about previous training examples

    Return:
        Number of examples configured
    """
    pngDir = os.path.abspath(pngDir) + '/'
    fCount = 0
    for f in os.listdir(pngDir):
        # the mask data files are added to the NN input when we consider the corresponding
        # so skip looking at the mask data files themselves
        if 'mask' in f:
            continue

        fCount += 1
        img = imgio.imread(pngDir + f)
        fileInfo = {}
        fileInfo['dbInfo'] = {'frameID': -1, 'vID': 'CS230'}
        fileInfo['width'] = img.shape[1]
        fileInfo['fpath_img'] = pngDir + f
        fileInfo['ade_scene'] = 'brain'
        fileInfo['height'] = img.shape[0]
        fileInfo['dbName'] = 'CS230'
        segFile = pngDir + f[:-4] + '_mask.png'
        if not os.path.isfile(segFile):
            raise RuntimeWarning('Mask corresponding to {} is missing'.format(pngDir + f))
        fileInfo['fpath_segm'] = segFile
    
        with open(jsonFile, 'a') as results:
            json.dump(fileInfo, results)
            results.write('\n')

    return fCount


def main(args):
    """
    Convert all patient data from DCM and NRRD formats to PNG
    """

    # Create dictionary mapping segmentation masks to CT Scans
    dict = util.seg_map_dict(args.mask_dir, args.dcm_root)

    # for each patient, convert the DICOM files and masks to png and save these in a new directory
    for key, value in dict.items():
        if value == '':
            continue
        patientName = os.path.basename(key)
        patientName = patientName.split('.')[0]
        outputDir = os.path.abspath(os.path.abspath(args.output_dir) + '/' + patientName)
        # import pdb; pdb.set_trace()
        convert_sample(value, key, outputDir)
        build_NN_inputs(outputDir, args.output_file)

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dcm_root', '-d', type=str, required=True,
                        help='Root directory containing patient directories. The patient directories contain DCM files for each patient.')
    parser.add_argument('--mask_dir', '-m', type=str, required=True,
                        help='Directory containing segmentation masks.')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help='Name of output directory containing PNG files created')
    parser.add_argument('--output_file', '-f', type=str, required=True,
                        help='Name of output file containing information about each sample to feed into semantic-segmentation-NN.')
    
    args = parser.parse_args()

    main(args)
