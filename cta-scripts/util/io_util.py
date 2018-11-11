import argparse
import pydicom as dicom
import numpy as np

from datetime import date, datetime
from sys import stderr


def args_to_list(csv, allow_empty, arg_type=int, allow_negative=True):
    """Convert comma-separated arguments to a list. Only take non-negative values."""
    arg_vals = [arg_type(d) for d in str(csv).split(',')]
    if not allow_negative:
        arg_vals = [v for v in arg_vals if v >= 0]
    if not allow_empty and len(arg_vals) == 0:
        return None
    return arg_vals


def print_err(*args, **kwargs):
    """Print a message to stderr."""
    print(*args, file=stderr, **kwargs)


def read_dicom(dicom_path):
    """Read a DICOM object from path to a DICOM.

    Args:
        dicom_path: Path to DICOM file to read.

    Raises:
        IOError: If we can't find a file at the path given.
    """
    dcm = None
    try:
        with open(dicom_path, 'rb') as dicom_file:
            dcm = dicom.dcmread(dicom_file)
    except IOError:
        print('Warning: Failed to open {}'.format(dicom_path))

    return dcm

def dcm_to_raw(dcm, dtype=np.int16):
    """Convert a DICOM object to a Numpy array of raw Housfield Units.

    Scale by the RescaleSlope, then add the RescaleIntercept (both DICOM header fields).
    
    Args:
        dcm: DICOM object
        dtype: Type of elements in output array

    Returns:
        ndarray of shape (height, width). Pixels are 'int16' raw Housfield Units.

    See Also:
        https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    """
    img_np = dcm.pixel_array
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    img_np = img_np.astype(dtype)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    img_np[img_np == - 2000] = 0

    intercept = dcm.RescaleIntercept
    slope = dcm.RescaleSlope

    if slope != 1:
        img_np = slope * img_np.astype(np.float64)
        img_np = img_np.astype(dtype)

    img_np += int(intercept)
    img_np = img_np.astype(np.int16)

    return img_np

def str_to_bool(v):
    """Convert an argument string into its boolean value."""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def json_encoder(obj):
    """JSON encoders for objects not normally supported by the JSON library."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))


def set_spawn_enabled():
    """Set PyTorch start method to spawn a new process rather than spinning up a new thread.

    This change was necessary to allow multiple DataLoader workers to read from an HDF5 file.

    See Also:
        https://github.com/pytorch/pytorch/issues/3492
    """
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass


def try_parse(s, type_fn=int):
    """Try parsing a string into type given by `type_fn`, and return None on ValueError."""
    i = None
    try:
        i = type_fn(s)
    except ValueError:
        pass
    return i
