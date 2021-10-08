"""Annotator utils."""

import progressbar


class DownloadProgressBar:
    """Progress bar."""

    def __init__(self):
        """Progress bar initializer."""
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        """Call."""
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(
                maxval=total_size if total_size > 0 else 1e-2
            )
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()
