"""
Alias entrypoint for DCWT-v2 100M benchmark.
"""

import os

from train_100m_bpe import main


if __name__ == "__main__":
    os.environ.setdefault("WFL_ACCEL_MODEL", "dcwt")
    main()
