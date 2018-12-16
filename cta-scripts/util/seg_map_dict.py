"""
Harry Emeric
CS230
"""

import os
import warnings

def seg_map_dict(seg_dir, scan_dir):
    
    """
    Take in a directory containing segmentation data for the patients, find the coresponding directory 
    with the original DICOM scans, and return a dictionary with the .nii.gz file as the key, and the
    corresponding directory of scans as the value.
    
    Arguments:
        seg_dir: [string] directory containing the segmentation data, 
            for example: '/data2/yeom/ky_aneur/segmentation/new_segmentation'
        scan_dir: [string] directory containing the original scans in DICOM
            format, for example '/data2/yeom/ky_aneur/sah/SAH_1.25/'
            
    Return:
        Dictionary mapping the segmentation data to the scans for each patient    
    """
    
    seg_dir = os.path.abspath(seg_dir) + '/'
    scan_dir = os.path.abspath(scan_dir) + '/'
    
    seg_map_dict = dict()

    for file in os.listdir(seg_dir):
        name = file.split('.')[0]
        path = ''
        if name in os.listdir(scan_dir):
            path = scan_dir + name
        else:
            for item in os.listdir(scan_dir):
                if item.startswith(name):
                    path = scan_dir + item
                
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith('.dcm'):
                    assert seg_map_dict.get(file) == None # If this fails there is more than one directory containing a .dcm file
                    path = os.path.join(root)
                        
        seg_map_dict[seg_dir + file] = path

    warnings.warn('{} Did not get mapped; check name mismatch'.format([key for key,val in seg_map_dict.items() if val=='']))

    return seg_map_dict
