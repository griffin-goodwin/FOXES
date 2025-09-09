import argparse
import logging
import os
import time
from datetime import datetime, timedelta
from os import cpu_count

import download_sdo as sdo
import flare_event_downloader as fed
import sxr_downloader as sxr

class FlareDownloadProcessor:
    def __init__(self, FlareEventDownloader, SDODownloader, SXRDownloader, flaring_data=True):
        """
        Initialize the FlareDownloadProcessor.
        This class is responsible for processing AIA flare downloads.
        """
        self.FlareEventDownloader = FlareEventDownloader
        self.SDODownloader = SDODownloader
        self.SXRDownloader = SXRDownloader
        self.flaring_data = flaring_data

    def process_download(self, time_before_start=timedelta(minutes=5), time_after_end=timedelta(minutes=0)):

        fl_events = self.FlareEventDownloader.download_events()
        print(fl_events)
        [os.makedirs(os.path.join(self.SDODownloader.ds_path, str(c)), exist_ok=True) for c in
         [94, 131, 171, 193, 211, 304]]
        
        # Create a progress file to track completed downloads
        progress_file = os.path.join(self.SDODownloader.ds_path, 'download_progress.txt')
        completed_dates = set()
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                completed_dates = set(line.strip() for line in f)
            print(f"Resuming from {len(completed_dates)} previously completed downloads")

        if self.flaring_data == True:
            print("Processing flare events...")
            if fl_events.empty:
                print("No flare events found. Skipping flare processing.")
                return
            for i, events in enumerate(fl_events.iterrows()):
                event = events[1]
                start_time = event['event_starttime'] - time_before_start
                end_time = event['event_endtime'] + time_after_end
                self.SXRDownloader.download_and_save_goes_data(start_time.strftime('%Y-%m-%d'),
                                                               end_time.strftime('%Y-%m-%d'), max_workers=2)
                processed_dates = set()
                for d in [start_time + i * timedelta(minutes=1) for i in
                          range((end_time - start_time) // timedelta(minutes=1))]:
                    # Only download if we haven't processed this date yet
                    if d.isoformat() not in processed_dates:
                        self.SDODownloader.downloadDate(d)
                        processed_dates.add(d.isoformat())
                logging.info(f"Processed flare event {i + 1}/{len(fl_events)}: {event['event_starttime']} to {event['event_endtime']}")
        elif self.flaring_data == False:
            print("Processing non-flare events...")
            if fl_events.empty or 'event_starttime' not in fl_events.columns:
                print("No flare events found or invalid data structure. Skipping non-flare processing.")
                return
            start_time_fl = fl_events['event_starttime']
            end_time_fl = fl_events['event_endtime']
            for i, events in enumerate(fl_events.iterrows()):
                start_time = end_time_fl[i]
                end_time = start_time_fl[i+1] if i + 1 < len(fl_events) else end_time_fl[-1] + timedelta(minutes=5)
                self.SXRDownloader.download_and_save_goes_data(start_time.strftime('%Y-%m-%d'),
                                                               end_time.strftime('%Y-%m-%d'), max_workers=2)
                
                # Adaptive sampling based on quiet period duration
                quiet_duration = end_time - start_time
                if quiet_duration < timedelta(hours=1):
                    # Short quiet period - sample every 10 minutes
                    sampling_interval = timedelta(minutes=5)
                elif quiet_duration < timedelta(days=1):
                    # Medium quiet period - sample every hour
                    sampling_interval = timedelta(minutes=30)
                else:
                    # Long quiet period - sample every 6 hours
                    sampling_interval = timedelta(hours=1)
                
                processed_dates = set()
                dates_to_process = [start_time + i * sampling_interval for i in
                                   range((end_time - start_time) // sampling_interval)]
                
                # Process in smaller batches to avoid overwhelming the server
                batch_size = 50  # Smaller batches
                for i in range(0, len(dates_to_process), batch_size):
                    batch = dates_to_process[i:i + batch_size]
                    print(f"Processing batch {i//batch_size + 1}/{(len(dates_to_process) + batch_size - 1)//batch_size} ({len(batch)} dates)")
                    
                    for j, d in enumerate(batch):
                        # Only download if we haven't processed this date yet
                        if d.isoformat() not in processed_dates and d.isoformat() not in completed_dates:
                            try:
                                print(f"  Downloading data for {d} ({j+1}/{len(batch)})")
                                self.SDODownloader.downloadDate(d)
                                processed_dates.add(d.isoformat())
                                completed_dates.add(d.isoformat())
                                
                                # Update progress file
                                with open(progress_file, 'a') as f:
                                    f.write(f"{d.isoformat()}\n")
                                
                                print(f"  ✓ Successfully downloaded {d}")
                                
                                # Add small delay between individual downloads
                                if j < len(batch) - 1:
                                    time.sleep(2)
                                    
                            except Exception as e:
                                print(f"  ✗ Failed to download data for {d}: {e}")
                                # If it's a connection error, wait longer before retrying
                                if "Connection refused" in str(e) or "timeout" in str(e).lower():
                                    print(f"  Waiting 10 seconds before continuing...")
                                    time.sleep(10)
                                continue
                        elif d.isoformat() in completed_dates:
                            print(f"  ⏭ Skipping {d} (already completed)")
                            processed_dates.add(d.isoformat())
                    
                    # Add longer delay between batches to avoid rate limiting
                    if i + batch_size < len(dates_to_process):
                        print("Waiting 10 seconds before next batch...")
                        time.sleep(10)
                logging.info(f"Processed non-flare event {i + 1}/{len(fl_events)}: {start_time} to {end_time} (duration: {quiet_duration}, samples: {len(processed_dates)})")
        else:
            raise ValueError("Invalid value for flaring_data. It should be either True or False.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download flare events and associated SDO data.')
    parser.add_argument('--start_date', type=str, default='2012-01-01',
                        help='Start date for downloading flare events (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2013-01-01',
                        help='End date for downloading flare events (YYYY-MM-DD)')
    parser.add_argument('--chunk_size', type=int, default=180,
                        help='Number of days per chunk for processing (default: 180)')
    parser.add_argument('--download_dir', type=str, default='/mnt/data',
                        help='Directory to save downloaded data (default: /mnt/data)')
    parser.add_argument('--flaring_data', dest='flaring_data', action='store_true',
                        help='Download flaring data (default)')
    parser.add_argument('--non_flaring_data', dest='non_flaring_data', action='store_false',
                        help='Download non-flaring data')
    parser.set_defaults(flaring_data=True)
    args = parser.parse_args()

    download_dir = args.download_dir
    start_date = args.start_date
    end_date = args.end_date
    chunk_size = args.chunk_size
    flaring_data = args.flaring_data
    # Parse start and end dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Process in chunks
    current_start = start
    while current_start < end:
        current_end = min(current_start + timedelta(days=chunk_size), end)
        print(f"Processing chunk: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")

        sxr_downloader = sxr.SXRDownloader(f"{download_dir}/GOES-flaring",
                                           f"{download_dir}/GOES-flaring/combined")
        flare_event = fed.FlareEventDownloader(
            current_start.strftime("%Y-%m-%d"),
            current_end.strftime("%Y-%m-%d"),
            event_type="FL",
            GOESCls="M1.0",
            directory=f"{download_dir}/SDO-AIA-flaring/FlareEvents"
        )
        sdo_downloader = sdo.SDODownloader(f"{download_dir}/SDO-AIA-flaring", "ggoodwin5@gsu.edu")

        processor = FlareDownloadProcessor(flare_event, sdo_downloader, sxr_downloader,
                                           flaring_data=flaring_data)
        processor.process_download()

        current_start = current_end