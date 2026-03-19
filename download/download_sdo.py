import argparse
import logging
import multiprocessing
import os
import time
from datetime import timedelta, datetime
from urllib import request
from urllib.error import URLError, HTTPError


import drms
import numpy as np
import pandas as pd
from astropy.io import fits
from sunpy.io._fits import header_to_fits
from sunpy.util import MetaDict
from astropy import units as u


class SDODownloader:
    """
    Class to download SDO data from JSOC.

    Args:
        base_path (str): Path to the directory where the downloaded data should be stored.
        email (str): Email address for JSOC registration.
        wavelengths (list): List of wavelengths to download.
        n_workers (int): Number of worker threads for parallel download.
    """
    def __init__(self, base_path='/mnt/data/PAPER/SDOData', email=None, wavelengths=['94', '131', '171', '193', '211', '304', '335'], n_workers=4, cadence=60):
        self.ds_path = base_path
        self.wavelengths = [str(wl) for wl in wavelengths]
        self.n_workers = n_workers
        #[os.makedirs(os.path.join(base_path, wl), exist_ok=True) for wl in self.wavelengths + ['6173']]
        [os.makedirs(os.path.join(base_path, wl), exist_ok=True) for wl in self.wavelengths]
        self.drms_client = drms.Client(email=email)
        self.cadence = cadence

    def download(self, sample):
        """
        Download the data from JSOC.

        Args:
            sample (tuple): Tuple containing the header, segment and time information.

        Returns:
            str: Path to the downloaded file.
        """
        header, segment, t = sample
        try:
            dir = os.path.join(self.ds_path, '%d' % header['WAVELNTH'])
            map_path = os.path.join(dir, '%s.fits' % t.isoformat('T', timespec='seconds'))
            if os.path.exists(map_path):
                return map_path
            # load map
            if not segment or pd.isna(segment):
                logging.error('Segment path is null for %s — data may not be in JSOC cache' % header.get('DATE__OBS'))
                raise ValueError('Null segment path for %s' % header.get('DATE__OBS'))
            url = 'http://jsoc.stanford.edu' + segment
            logging.info('Downloading: %s' % url)
            
            # Retry download with exponential backoff
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Create a custom opener with timeout
                    opener = request.build_opener()
                    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (compatible; SDO-Downloader/1.0)')]
                    request.install_opener(opener)
                    
                    # Download with timeout
                    request.urlretrieve(url, filename=map_path)
                    break
                    
                except (URLError, HTTPError) as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                        logging.warning(f'Download attempt {attempt + 1} failed for {url}: {e}')
                        logging.warning(f'Retrying in {wait_time} seconds...')
                        time.sleep(wait_time)
                        continue
                    else:
                        logging.error(f'All download attempts failed for {url}: {e}')
                        raise e
                except Exception as e:
                    logging.error(f'Unexpected error downloading {url}: {e}')
                    raise e

            header['DATE_OBS'] = header['DATE__OBS']
            header = header_to_fits(MetaDict(header))
            with fits.open(map_path, 'update') as f:
                hdr = f[1].header
                for k, v in header.items():
                    if pd.isna(v):
                        continue
                    hdr[k] = v
                f.verify('silentfix')

            return map_path
        except Exception as ex:
            logging.info('Download failed: %s (requeue)' % header['DATE__OBS'])
            logging.info(ex)
            raise ex

    def downloadDate(self, date):
        """
        Download the data for the given date.

        Args:
            date (datetime): The date for which the data should be downloaded.

        Returns:
            list: List of paths to the downloaded files.
        """
        id = date.isoformat()

        logging.info('Start download: %s' % id)

        # query EUV
        time_param = '%sZ' % date.isoformat('_', timespec='seconds')
        ds_euv = 'aia.lev1_euv_12s[%s][%s]{image}' % (time_param, ','.join(self.wavelengths))
        keys_euv = self.drms_client.keys(ds_euv)
        header_euv, segment_euv = self.drms_client.query(ds_euv, key=','.join(keys_euv), seg='image')
        logging.info('Fast-path query returned %d rows (need %d), qualities: %s' % (
            len(header_euv), len(self.wavelengths),
            list(header_euv.QUALITY) if len(header_euv) > 0 else []))
        if len(header_euv) != len(self.wavelengths) or np.any(header_euv.QUALITY.fillna(0) != 0):
            self.fetchDataFallback(date)
            return

        queue = []
        for (idx, h), s in zip(header_euv.iterrows(), segment_euv.image):
            queue += [(h.to_dict(), s, date)]

        with multiprocessing.Pool(self.n_workers) as p:
            p.map(self.download, queue)
        
        # Add a small delay to be respectful to the server
        time.sleep(.1)
        logging.info('Finished: %s' % id)

    def fetchDataFallback(self, date):
        """
        Download the data for the given date using fallback.

        Args:
            date (datetime): The date for which the data should be downloaded.

        Returns:
            list: List of paths to the downloaded files.
        """
        id = date.isoformat()

        logging.info('Fallback download: %s' % id)
        header_euv, segment_euv = [], []
        t = date - timedelta(hours=6)
        for wl in self.wavelengths:
            euv_ds = 'aia.lev1_euv_12s[%sZ/12h@12s][%s]{image}' % (
                t.replace(tzinfo=None).isoformat('_', timespec='seconds'), wl)
            keys_euv = self.drms_client.keys(euv_ds)
            header_tmp, segment_tmp = self.drms_client.query(euv_ds, key=','.join(keys_euv), seg='image')
            logging.info('Fallback query wl=%s returned %d rows' % (wl, len(header_tmp)))
            assert len(header_tmp) != 0, 'No data found for wl=%s at %s' % (wl, id)
            date_str = header_tmp['DATE__OBS'].replace('MISSING', '').str.replace('60', '59')  # fix date format
            date_diff = (pd.to_datetime(date_str).dt.tz_localize(None) - date).abs()
            # sort and filter
            header_tmp['date_diff'] = date_diff
            segment_tmp['date_diff'] = date_diff
            cond_tmp = (header_tmp.QUALITY == 0) | header_tmp.QUALITY.isna()
            header_filtered = header_tmp[cond_tmp]
            segment_filtered = segment_tmp[cond_tmp]
            if len(header_filtered) > 0:
                header_tmp = header_filtered
                segment_tmp = segment_filtered
            else:
                logging.warning('No quality-0 EUV frames for wl=%s at %s — using closest available' % (wl, id))
            header_euv.append(header_tmp.sort_values('date_diff').iloc[0].drop('date_diff'))
            segment_euv.append(segment_tmp.sort_values('date_diff').iloc[0].drop('date_diff'))

        queue = []
        #queue += [(header_hmi.to_dict(), segment_hmi.magnetogram, date)]
        for h, s in zip(header_euv, segment_euv):
            queue += [(h.to_dict(), s.image, date)]

        with multiprocessing.Pool(self.n_workers) as p:
            p.map(self.download, queue)

        # Add a small delay to be respectful to the server
        time.sleep(.1)
        logging.info('Finished: %s' % id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download SDO data from JSOC with quality check and fallback')
    parser.add_argument('--download_dir', type=str, help='path to the download directory.')
    parser.add_argument('--email', type=str, help='registered email address for JSOC.')
    parser.add_argument('--start_date', type=str, help='start date in format YYYY-MM-DD.')
    parser.add_argument('--end_date', type=str, help='end date in format YYYY-MM-DD HH:MM:SS.', required=False,
                        default=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    parser.add_argument('--cadence', type=int, help='cadence in minutes.', required=False, default=60)

    args = parser.parse_args()
    download_dir = args.download_dir
    start_date = args.start_date
    end_date = args.end_date
    cadence = args.cadence

    [os.makedirs(os.path.join(download_dir, str(c)), exist_ok=True) for c in [94, 131, 171, 193, 211, 304, 335]]
    downloader = SDODownloader(base_path=download_dir, email=args.email)
    start_date_datetime = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    #end_date = datetime.now()
    end_date_datetime = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")


    #Skip over dates that already exist in the download directory
    for d in [start_date_datetime + i * timedelta(minutes=cadence) for i in
              range((end_date_datetime - start_date_datetime) // timedelta(minutes=cadence))]:
        #make sure the file exists in all wavelengths directories
        for wl in [94, 131, 171, 193, 211, 304, 335]:
            if not os.path.exists(os.path.join(
                download_dir, 
                str(wl), 
                f"{d.year:04d}-{d.month:02d}-{d.day:02d}T{d.hour:02d}:{d.minute:02d}:{d.second:02d}.fits"
            )):
                break
        else:
            logging.info(f"Skipping {d.isoformat()} because it already exists in the download directory")
            continue
        downloader.downloadDate(d)
