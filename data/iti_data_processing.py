import collections.abc

collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
# Now import hyper
import numpy as np
from astropy.visualization import ImageNormalize, AsinhStretch
from itipy.data.dataset import StackDataset, get_intersecting_files, AIADataset
from itipy.data.editor import BrightestPixelPatchEditor, sdo_norms
import os
import json
from multiprocessing import Pool
from tqdm import tqdm


def load_config():
    """Load configuration from environment or use defaults."""
    try:
        config = json.loads(os.environ['PIPELINE_CONFIG'])
        return config
    except:
        pass



class SDODataset_flaring(StackDataset):
    """
    Dataset for SDO data

    Args:
        data: Data
        patch_shape (tuple): Patch shape
        wavelengths (list): List of wavelengths
        resolution (int): Resolution
        ext (str): File extension
        **kwargs: Additional arguments
    """

    def __init__(self, data, patch_shape=None, wavelengths=None, resolution=2048, ext='.fits', allow_errors=False, **kwargs):
        if isinstance(data, list):
            paths = data
        else:
            paths = get_intersecting_files(data, wavelengths, ext=ext, **kwargs)
        ds_mapping = {94: AIADataset, 131: AIADataset, 171: AIADataset, 193: AIADataset, 211: AIADataset,
                      304: AIADataset, 335: AIADataset, 1600: AIADataset, 1700: AIADataset, 4500: AIADataset, 6173: AIADataset}
        data_sets = [ds_mapping[wl_id](files, wavelength=wl_id, resolution=resolution, ext=ext, allow_errors=allow_errors)
                     for wl_id, files in zip(wavelengths, paths)]
        super().__init__(data_sets, **kwargs)
        if patch_shape is not None:
            self.addEditor(BrightestPixelPatchEditor(patch_shape))


_aia_dataset = None
_output_folder = None


def _init_worker(dataset, out_folder):
    global _aia_dataset, _output_folder
    _aia_dataset = dataset
    _output_folder = out_folder


def save_sample(i):
    try:
        data = _aia_dataset[i]
        file_path = os.path.join(_output_folder, _aia_dataset.getId(i)) + '.npy'
        np.save(file_path, data)
    except Exception as e:
        print(f"Warning: Could not process sample {i} (ID: {_aia_dataset.getId(i)}): {e}")


def check_existing_files(base_input_folder, wavelengths, output_folder):
    """Check how many files already exist without loading the full dataset."""
    files = get_intersecting_files(base_input_folder, wavelengths, ext='.fits')
    if not files or len(files) == 0:
        return 0, 0

    existing_count = 0
    total_expected = len(files[0])

    for i in range(total_expected):
        first_wl_file = files[0][i]
        base_name = os.path.splitext(os.path.basename(first_wl_file))[0]
        if '_' in base_name:
            base_name = '_'.join(base_name.split('_')[:-1])
        output_path = os.path.join(output_folder, base_name) + '.npy'
        if os.path.exists(output_path):
            existing_count += 1

    return existing_count, total_expected


if __name__ == '__main__':
    config = load_config()
    wavelengths = config['iti']['wavelengths']
    base_input_folder = config['iti']['input_folder']
    output_folder = config['iti']['output_folder']
    os.makedirs(output_folder, exist_ok=True)

    existing_files, total_expected = check_existing_files(base_input_folder, wavelengths, output_folder)
    print(f"Found {existing_files} existing files out of {total_expected} expected files")

    if existing_files >= total_expected:
        print("All files already processed. Nothing to do.")
    else:
        print(f"Need to process {total_expected - existing_files} remaining files")

        aia_dataset = SDODataset_flaring(data=base_input_folder, wavelengths=wavelengths, resolution=512, allow_errors=True)

        unprocessed_indices = [
            i for i in range(len(aia_dataset))
            if not os.path.exists(os.path.join(output_folder, aia_dataset.getId(i)) + '.npy')
        ]
        print(f"Processing {len(unprocessed_indices)} unprocessed samples")

        if unprocessed_indices:
            with Pool(processes=os.cpu_count(), initializer=_init_worker, initargs=(aia_dataset, output_folder)) as pool:
                list(tqdm(pool.imap(save_sample, unprocessed_indices), total=len(unprocessed_indices)))
            print("AIA data processing completed.")
        else:
            print("All samples already processed. Nothing to do.")
