import argparse
import logging
import os
import shutil
import socket
import time
from datetime import timedelta, datetime
from multiprocessing.pool import ThreadPool
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
        tolerance (int): Hard half-width, in seconds, of the search window around the
            requested timestamp. Only frames within [t-tolerance, t+tolerance] are
            ever considered; the nearest quality-0 frame per wavelength is selected.
            A wavelength with no frame in that window is reported missing rather than
            substituting an unrelated image far from the peak. Default 120s.
        timeout (int): Per-file HTTP download timeout in seconds.
    """
    def __init__(self, base_path='/mnt/data/PAPER/SDOData', email=None,
                 wavelengths=['94', '131', '171', '193', '211', '304', '335'],
                 n_workers=4, cadence=60, tolerance=120, timeout=120):
        self.ds_path = base_path
        self.wavelengths = [str(wl) for wl in wavelengths]
        self.n_workers = n_workers
        self.tolerance = tolerance
        self.timeout = timeout
        [os.makedirs(os.path.join(base_path, wl), exist_ok=True) for wl in self.wavelengths]
        self.drms_client = drms.Client(email=email)
        self.cadence = cadence
        self._keys_cache = None

    # ------------------------------------------------------------------ helpers

    def _keys_str(self):
        """Comma-joined keyword list for the EUV series, fetched once and cached.

        The keyword set is identical for every record, so querying it per-timestamp
        (as the old code did) was a wasted JSOC round-trip on every download.
        """
        if self._keys_cache is None:
            self._keys_cache = ','.join(self.drms_client.keys('aia.lev1_euv_12s'))
        return self._keys_cache

    def _query_window(self, date, half_seconds):
        """Query all wavelengths over [date-half, date+half] in a single JSOC call.

        Returns the (header, segment) DataFrames as produced by drms.
        """
        start = date - timedelta(seconds=half_seconds)
        dur = 2 * half_seconds
        ds = 'aia.lev1_euv_12s[%sZ/%ds@12s][%s]{image}' % (
            start.isoformat('_', timespec='seconds'), dur, ','.join(self.wavelengths))
        return self.drms_client.query(ds, key=self._keys_str(), seg='image')

    def _select_nearest(self, header, segment, date, wavelengths=None):
        """Pick the nearest quality-0 frame per wavelength from a query result.

        Returns a list of pandas Series (one per wavelength that had any frame),
        each carrying all queried keywords plus '_seg' (the segment path) and
        '_diff' (time offset from `date`). Wavelengths with no rows are skipped.
        """
        wavelengths = wavelengths if wavelengths is not None else self.wavelengths
        if len(header) == 0:
            return []
        header = header.copy()
        header['_seg'] = segment['image'].values
        # DATE__OBS occasionally carries 'MISSING' or a seconds=60 artifact; clean both.
        date_str = (header['DATE__OBS'].astype(str)
                    .str.replace('MISSING', '', regex=False)
                    .str.replace('60', '59', regex=False))
        obs = pd.to_datetime(date_str, errors='coerce', utc=True).dt.tz_localize(None)
        header['_diff'] = (obs - date).abs()

        rows = []
        for wl in wavelengths:
            sub = header[header['WAVELNTH'] == int(wl)]
            if len(sub) == 0:
                continue
            good = sub[(sub['QUALITY'] == 0) | sub['QUALITY'].isna()]
            if len(good) > 0:
                sub = good
            else:
                logging.warning('No quality-0 frame for wl=%s near %s — using closest available'
                                % (wl, date.isoformat()))
            rows.append(sub.sort_values('_diff').iloc[0])
        return rows

    def _fetch_rows(self, date):
        """Resolve the nearest valid frame per wavelength within +-tolerance of `date`.

        The query window is bounded by `self.tolerance`, so a frame far from the
        peak can never be returned: a wavelength with no frame in the window is
        reported missing (recorded downstream) rather than backfilled with an
        unrelated image. This is deliberate for flare work — a frame hours from
        the peak is scientifically wrong, not a usable substitute.
        """
        header, segment = self._query_window(date, self.tolerance)
        rows = self._select_nearest(header, segment, date)
        have = {int(r['WAVELNTH']) for r in rows}
        missing = [wl for wl in self.wavelengths if int(wl) not in have]
        if missing:
            logging.warning('%s: no frame within +-%ds for wavelengths %s — reported missing'
                            % (date.isoformat(), self.tolerance, missing))
        return rows

    # ------------------------------------------------------------------ download

    def download(self, sample):
        """
        Download a single segment from JSOC to a FITS file (atomic, with retries).

        Args:
            sample (tuple): (header dict, segment path, time).

        Returns:
            str: Path to the downloaded file.
        """
        header, segment, t = sample
        try:
            dir = os.path.join(self.ds_path, '%d' % header['WAVELNTH'])
            map_path = os.path.join(dir, '%s.fits' % t.isoformat('T', timespec='seconds'))
            if os.path.exists(map_path):
                return map_path
            if not segment or pd.isna(segment):
                logging.error('Segment path is null for %s — data may not be in JSOC cache'
                              % header.get('DATE__OBS'))
                raise ValueError('Null segment path for %s' % header.get('DATE__OBS'))
            url = 'http://jsoc.stanford.edu' + segment
            logging.info('Downloading: %s' % url)

            # Download to a temp file so a crash/partial transfer never leaves a
            # truncated .fits that the resume logic would mistake for "done".
            tmp_path = map_path + '.part'
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    req = request.Request(
                        url, headers={'User-Agent': 'Mozilla/5.0 (compatible; SDO-Downloader/1.0)'})
                    with request.urlopen(req, timeout=self.timeout) as resp, open(tmp_path, 'wb') as f:
                        shutil.copyfileobj(resp, f)
                    break
                except (URLError, HTTPError, socket.timeout, TimeoutError) as e:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                        logging.warning('Download attempt %d failed for %s: %s' % (attempt + 1, url, e))
                        logging.warning('Retrying in %d seconds...' % wait_time)
                        time.sleep(wait_time)
                        continue
                    logging.error('All download attempts failed for %s: %s' % (url, e))
                    raise

            header['DATE_OBS'] = header['DATE__OBS']
            header = header_to_fits(MetaDict(header))
            with fits.open(tmp_path, 'update') as f:
                hdr = f[1].header
                for k, v in header.items():
                    if pd.isna(v):
                        continue
                    hdr[k] = v
                f.verify('silentfix')

            os.replace(tmp_path, map_path)
            return map_path
        except Exception as ex:
            logging.info('Download failed: %s (requeue)' % header.get('DATE__OBS'))
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

        rows = self._fetch_rows(date)
        if not rows:
            logging.warning('No frames found for %s' % id)
            return

        queue = [(r.drop(['_seg', '_diff']).to_dict(), r['_seg'], date) for r in rows]

        # Downloads are I/O bound (urlretrieve); threads avoid per-timestamp
        # process-pool spawn overhead and share the cached keyword list.
        with ThreadPool(self.n_workers) as p:
            p.map(self.download, queue)

        time.sleep(.1)  # be gentle with the server
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
