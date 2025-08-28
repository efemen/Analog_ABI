# h5_mat_logger.py
from __future__ import annotations

import os
import time
import math
from datetime import datetime
from typing import Iterable, Optional, Dict, Any

import numpy as np
import h5py

# Optional dependencies for MATLAB export
try:
    from scipy.io import savemat  # for MAT v7
except Exception:
    savemat = None



class H5MatLogger:
    """
    Stream AI (multi-channel) + encoder angle (degrees) to HDF5 efficiently,
    then convert to MATLAB .mat on demand.

    Datasets created:
      - /ai            : shape (num_ai_ch, N), float64
      - /encoder_deg   : shape (N,), float64
      - /time          : shape (N,), float64
    File attributes include:
      - ai_rate_hz, device, ai_channels (list of str), ppr, created_utc, notes

    Usage:
        logger = H5MatLogger(path='run.h5', num_ai_ch=7, chunk_len=2000,
                             ai_rate=20000, device='Dev2',
                             ai_channel_names=[f'ai{i}' for i in range(1,8)],
                             ppr=1000)

        # during acquisition loop (ai_block: (7, CHUNK), enc_block: (CHUNK,))
        logger.write(ai_block, enc_block)

        # when done:
        logger.close()
        logger.to_mat('run_v73.mat', version='v7.3')   # or version='v7'
    """

    def __init__(
        self,
        path: Optional[str],
        num_ai_ch: int,
        chunk_len: int,
        ai_rate: float,
        device: str,
        ai_channel_names: Optional[Iterable[str]] = None,
        ppr: Optional[int] = None,
        notes: Optional[str] = None,
        compression: Optional[str] = None,      # e.g., "gzip" for smaller files (slower); None is fastest
        flush_every: int = 10,                  # flush HDF5 every N writes
        make_dirs: bool = True,
    ) -> None:
        if path is None:
            path = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.h5"

        self.path = os.path.abspath(path)
        if make_dirs:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

        self.num_ai_ch = int(num_ai_ch)
        self.chunk_len = int(chunk_len)
        self.ai_rate = float(ai_rate)
        self.device = str(device)
        self.ppr = int(ppr) if ppr is not None else None
        self.notes = str(notes) if notes else ""
        self.compression = compression
        self.flush_every = int(flush_every)
        self._writes_since_flush = 0
        self._closed = False

        if ai_channel_names is None:
            self.ai_channel_names = [f"ai{i}" for i in range(self.num_ai_ch)]
        else:
            self.ai_channel_names = list(ai_channel_names)
            if len(self.ai_channel_names) != self.num_ai_ch:
                raise ValueError("len(ai_channel_names) must equal num_ai_ch")

        # Open HDF5 and create extendable datasets
        self.f = h5py.File(self.path, "w")

        # Metadata / attributes at file root
        self.f.attrs["created_utc"] = datetime.utcnow().isoformat() + "Z"
        self.f.attrs["ai_rate_hz"] = self.ai_rate
        self.f.attrs["device"] = self.device
        self.f.attrs["ai_channels"] = np.array(self.ai_channel_names, dtype=h5py.string_dtype())
        if self.ppr is not None:
            self.f.attrs["ppr"] = self.ppr
        if self.notes:
            self.f.attrs["notes"] = self.notes

        # Save start time
        self.start_time = time.time()
        self.f.attrs["start_time"] = self.start_time

        # Datasets: extendable along sample axis
        # chunks set to (num_ai_ch, chunk_len) and (chunk_len,) for fast appends
        self.ds_ai = self.f.create_dataset(
            "ai",
            shape=(self.num_ai_ch, 0),
            maxshape=(self.num_ai_ch, None),
            chunks=(self.num_ai_ch, self.chunk_len),
            dtype="f8",
            compression=self.compression,
        )
        self.ds_enc = self.f.create_dataset(
            "encoder_deg",
            shape=(0,),
            maxshape=(None,),
            chunks=(self.chunk_len,),
            dtype="f8",
            compression=self.compression,
        )
        self.ds_time = self.f.create_dataset(
            "time",
            shape=(0,),
            maxshape=(None,),
            chunks=(self.chunk_len,),
            dtype="f8",
            compression=self.compression,
        )

        # Running sample counter
        self.n_total = 0

    # ---------------------- Streaming API ----------------------

    def write(self, ai_block: np.ndarray, enc_block: np.ndarray) -> None:
        """
        Append a matched chunk to the HDF5 datasets.

        ai_block:  shape (num_ai_ch, N)
        enc_block: shape (N,)
        """
        if self._closed:
            raise RuntimeError("Logger is closed.")

        ai_block = np.asarray(ai_block, dtype=np.float64)
        enc_block = np.asarray(enc_block, dtype=np.float64)

        if ai_block.ndim == 1:
            # If a single sample per channel came as (num_ai_ch,), promote to (num_ai_ch, 1)
            ai_block = ai_block.reshape(self.num_ai_ch, 1)
        if enc_block.ndim == 0:
            enc_block = enc_block.reshape(1)
        if ai_block.shape[0] != self.num_ai_ch:
            raise ValueError(f"ai_block has {ai_block.shape[0]} channels, expected {self.num_ai_ch}")
        if ai_block.shape[1] != enc_block.shape[0]:
            raise ValueError("ai_block and enc_block must have the same sample count")

        N = ai_block.shape[1]
        new_n = self.n_total + N

        # Resize and write
        self.ds_ai.resize((self.num_ai_ch, new_n))
        self.ds_ai[:, self.n_total:new_n] = ai_block

        self.ds_enc.resize((new_n,))
        self.ds_enc[self.n_total:new_n] = enc_block

        # Calculate and write time array
        times = np.arange(self.n_total, new_n, dtype=np.float64) / self.ai_rate + self.start_time
        self.ds_time.resize((new_n,))
        self.ds_time[self.n_total:new_n] = times

        self.n_total = new_n

        # Periodic flush for safety
        self._writes_since_flush += 1
        if self.flush_every > 0 and self._writes_since_flush >= self.flush_every:
            self.f.flush()
            self._writes_since_flush = 0

    def write_attrs(self, extra: Dict[str, Any]) -> None:
        """Attach additional metadata (attributes) to the file root."""
        for k, v in extra.items():
            # store strings safely
            if isinstance(v, str):
                self.f.attrs[k] = v
            else:
                self.f.attrs[k] = np.array(v)

    @property
    def seconds_logged(self) -> float:
        """Total time represented by samples written."""
        return float(self.n_total) / self.ai_rate if self.ai_rate > 0 else math.nan

    # ---------------------- Close / Convert ----------------------

    def close(self) -> None:
        if not self._closed:
            try:
                self.f.flush()
            finally:
                self.f.close()
            self._closed = True

    def to_mat(self, out_path: Optional[str] = None, version: str = "v7.3") -> str:
        """
        Convert the stored data to a .mat file AFTER acquisition.
        version: 'v7.3' (HDF5-based; requires hdf5storage) or 'v7' (requires scipy).
        Returns the output path.
        NOTE: Both conversion paths load arrays in memory once. For very large files,
              prefer using the original HDF5 in MATLAB via h5read().
        """
        if not self._closed:
            # Be safe: flush before reading back
            self.f.flush()

        h5_path = self.path
        if out_path is None:
            base, _ = os.path.splitext(h5_path)
            out_path = base + ("_v73.mat" if version.lower() == "v7.3" else "_v7.mat")

        with h5py.File(h5_path, "r") as f:
            mdict = {
                "ai": f["ai"][...],                       # (num_ai_ch, N)
                "encoder_deg": f["encoder_deg"][...],     # (N,)
                "time": f["time"][...],                   # (N,)
                "ai_rate_hz": float(f.attrs.get("ai_rate_hz", np.nan)),
                "device": _maybe_to_str(f.attrs.get("device", "")),
                "ai_channels": np.array(
                    [_maybe_to_str(s) for s in f.attrs.get("ai_channels", [])],
                    dtype=object
                ),
            }
            if "ppr" in f.attrs:
                mdict["ppr"] = int(f.attrs["ppr"])
            if "notes" in f.attrs:
                mdict["notes"] = _maybe_to_str(f.attrs["notes"])
            if "created_utc" in f.attrs:
                mdict["created_utc"] = _maybe_to_str(f.attrs["created_utc"])

        savemat(out_path, mdict, do_compression=True)

        return os.path.abspath(out_path)

    # ---------------------- Context manager ----------------------

    def __enter__(self) -> "H5MatLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _maybe_to_str(x):
    """Convert HDF5 attributes (which may be bytes, numpy scalars) to plain str."""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    if hasattr(x, "astype") and getattr(x, "dtype", None) is not None:
        try:
            return str(x.astype(str))
        except Exception:
            return str(x)
    return str(x)
