import argparse
import h5py
import json
import numpy as np
import os
import pickle
import util


from tqdm import tqdm


def add_to_hdf5(series_list, output_dir, max_series=1e5):
    # Open HDF5 file for appending
    hdf5_fh = h5py.File(os.path.join(output_dir, 'data.hdf5'), 'a')
    for group_name in ('series', 'aneurysm_masks'):
        if group_name not in hdf5_fh:
            hdf5_fh.create_group('/{}'.format(group_name))

    assert len(series_list) < 1e5, 'Too many series for 5-digit IDs.'
    for i, s in enumerate(series_list):
        if i >= max_series:
            break
        dset_path = '/series/{:05d}'.format(i+1)
        if dset_path in hdf5_fh:
            continue
        print('Processing series {} from study {}...'.format(s.series_number, s.study_name))
        pixel_arrays = []
        is_valid_series = True
        for slice_name in tqdm(s.slice_names, total=len(s), unit=' slices'):
            # Process and write slices
            dcm_path = os.path.join(s.dcm_dir, slice_name + '.dcm')
            dcm = util.read_dicom(dcm_path)
            try:
                pixel_arrays.append(util.dcm_to_raw(dcm))
            except NotImplementedError:
                print('Unsupported image format, not converting study: {}'.format(s.study_name))
                is_valid_series = False
                break
        if not is_valid_series:
            continue

        volume = np.stack(pixel_arrays)

        aneurysm_mask_path = os.path.join(s.dcm_dir, 'aneurysm_mask.npy')
        if os.path.exists(aneurysm_mask_path):
            s.aneurysm_mask_path = aneurysm_mask_path
            aneurysm_mask = np.transpose(np.load(s.aneurysm_mask_path), [2, 0, 1])
        else:
            s.aneurysm_mask_path = None
            aneurysm_mask = None

        assert aneurysm_mask is None or aneurysm_mask.shape == volume.shape, \
            'Mismatched aneurysm mask and volume shapes: {} and {}'.format(aneurysm_mask.shape, volume.shape)

        # Create one dataset for the volume (int16), one for the mask (bool)
        s.dset_path = dset_path
        hdf5_fh.create_dataset(s.dset_path, data=volume, dtype='i2', chunks=True)

        if aneurysm_mask is not None:
            s.aneurysm_mask_path = '/aneurysm_masks/{:05d}'.format(i+1)
            hdf5_fh.create_dataset(s.aneurysm_mask_path, data=aneurysm_mask, dtype='?', chunks=True)

    # Print summary
    util.print_err('Series: {}'.format(len(hdf5_fh['/series'])))
    util.print_err('Aneurysm Masks: {}'.format(len(hdf5_fh['/aneurysm_masks'])))

    # Dump pickle and JSON (updated dset_path and mask_path attributes)
    util.print_err('Dumping pickle file...')
    with open(os.path.join(output_dir, 'series_list.pkl'), 'wb') as pkl_fh:
        pickle.dump(series_list, pkl_fh)
    util.print_err('Dumping JSON file...')
    with open(os.path.join(output_dir, 'series_list.json'), 'w') as json_file:
        json.dump([dict(series) for series in series_list], json_file,
                  indent=4, sort_keys=True, default=util.json_encoder)

    # Clean up
    hdf5_fh.close()


def get_aneurysm_range(mask):
    """Get range of slice indices where an aneurysm lives."""
    is_aneurysm = np.any(mask, axis=(1, 2))
    slice_min, slice_max = np.where(is_aneurysm)[0][[0, -1]]

    return [int(slice_min), int(slice_max)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--pkl_path', type=str, required=True,
                        help='Path to pickle file for a study.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output directory for JSON and Pickle files.')
    args_ = parser.parse_args()

    with open(args_.pkl_path, 'rb') as pkl_file:
        all_series = pickle.load(pkl_file)

    # add_to_hdf5(all_series, args_.output_dir, args_.max_series)
    add_to_hdf5(all_series, args_.output_dir)
