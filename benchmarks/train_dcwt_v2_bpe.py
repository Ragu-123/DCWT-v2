"""
Alias entrypoint for DCWT-v2 BPE benchmark.
"""

import os

from train_wave_v35_bpe import main


if __name__ == "__main__":
    os.environ.setdefault("WFL_ACCEL_MODEL", "dcwt")
    main()
