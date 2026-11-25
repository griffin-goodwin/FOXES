"""SunPy downloads with an explicitly owned asyncio loop."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from sunpy.util.parfive_helpers import Downloader


class ManagedDownloader(Downloader):
    """Close download loops before returning, including inside notebooks.

    Parfive 2.3.1's synchronous wrapper leaves its loop and signal handlers
    alive. Later collection in an AIA worker can then try to remove those
    handlers outside the main thread. Run the public download coroutine with
    ``asyncio.run`` so its loop is closed in the thread that owns it.
    """

    def download(self):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            running_loop = False
        else:
            running_loop = True

        if not running_loop:
            return self._download()

        # A notebook already owns this thread's running loop. Create and close
        # the download loop in a worker without installing signal handlers.
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(self._download).result()

    def _download(self):
        return asyncio.run(self.run_download())
