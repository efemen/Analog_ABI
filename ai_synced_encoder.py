import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType, Edge, TriggerType
from nidaqmx.stream_readers import AnalogMultiChannelReader, CounterReader
from nidaqmx.types import CtrFreq
from nidaqmx import constants
from nidaqmx.constants import AcquisitionType, EncoderType, AngleUnits, EncoderZIndexPhase
from h5_mat_logger import H5MatLogger

import time

# ----------------- Config -----------------
DEVICE = "Dev2"
COUNTER = "ctr0"
PPR = 1000

AI_RATE = 20_000           # Hz
AI_FIRST_CH = 1
AI_LAST_CH  = 7            # ai1:7 inclusive -> 7 channels
NUM_AI_CH = AI_LAST_CH - AI_FIRST_CH + 1

CHUNK = 2000               # samples per read (0.1 s at 20 kHz)
BUFFER_SECS = 10           # circular buffer length (seconds)
BUF_LEN = AI_RATE * BUFFER_SECS  # samples per channel in the ring buffer

logger = H5MatLogger(
    path=None,                    # auto timestamped: run_YYYYMMDD_HHMMSS.h5
    num_ai_ch=NUM_AI_CH,          # 7 for ai1:7
    chunk_len=CHUNK,              # same as your read chunk
    ai_rate=AI_RATE,              # 20000
    device=DEVICE,                # "Dev2"
    ai_channel_names=[f"ai{i}" for i in range(AI_FIRST_CH, AI_LAST_CH+1)],
    ppr=PPR,
    notes="ai1:7 @ 20 kHz + synced encoder"
)

# ------------- Encoder task (synced) -------------
task = nidaqmx.Task()
enc_ch = task.ci_channels.add_ci_ang_encoder_chan(
    counter=f"{DEVICE}/{COUNTER}",
    decoding_type=EncoderType.X_2,
    zidx_enable=True,                     # reset at index
    zidx_val=0.0,                         # reset angle to 0.0 on Z
    units=AngleUnits.DEGREES,
    zidx_phase=EncoderZIndexPhase.AHIGH_BHIGH,
    pulses_per_rev=PPR
)

# Terminals and filters as you had them
enc_ch.ci_encoder_a_input_term = f"/{DEVICE}/PFI8"
enc_ch.ci_encoder_b_input_term = f"/{DEVICE}/PFI9"
enc_ch.ci_encoder_z_input_term = f"/{DEVICE}/PFI10"

enc_ch.ci_encoder_a_input_dig_fltr_enable = True
enc_ch.ci_encoder_b_input_dig_fltr_enable = True
enc_ch.ci_encoder_z_input_dig_fltr_enable = True

enc_ch.ci_encoder_a_input_dig_fltr_min_pulse_width = 2e-6
enc_ch.ci_encoder_b_input_dig_fltr_min_pulse_width = 2e-6
enc_ch.ci_encoder_z_input_dig_fltr_min_pulse_width = 2e-6

ai_task = nidaqmx.Task()
ai_task.ai_channels.add_ai_voltage_chan(f"{DEVICE}/ai{AI_FIRST_CH}:{AI_LAST_CH}")
ai_task.timing.cfg_samp_clk_timing(
    rate=AI_RATE,
    sample_mode=AcquisitionType.CONTINUOUS,
    samps_per_chan=BUF_LEN  # host-side buffer size hint
)

# Encoder task uses the AI SampleClock and StartTrigger for perfect sync
task.timing.cfg_samp_clk_timing(
    rate=AI_RATE,
    source=f"/{DEVICE}/ai/SampleClock",
    sample_mode=AcquisitionType.CONTINUOUS,
    samps_per_chan=BUF_LEN
)

# Use AI StartTrigger to arm the counter task
arm = task.triggers.arm_start_trigger
arm.trig_type = TriggerType.DIGITAL_EDGE
arm.dig_edge_src = f"/{DEVICE}/ai/StartTrigger"
arm.dig_edge_edge = Edge.RISING

# ------------- Stream readers & circular buffers -------------
ai_reader = AnalogMultiChannelReader(ai_task.in_stream)
ctr_reader = CounterReader(task.in_stream)

# Chunk buffers for each read
_ai_chunk = np.empty((NUM_AI_CH, CHUNK), dtype=np.float64)
_enc_chunk = np.empty(CHUNK, dtype=np.float64)

# Circular buffers (last BUFFER_SECS seconds)
ai_ring = np.empty((NUM_AI_CH, BUF_LEN), dtype=np.float64)
enc_ring = np.empty(BUF_LEN, dtype=np.float64)
write_idx = 0
ring_filled = False

def _ring_write(ai_block: np.ndarray, enc_block: np.ndarray):
    """Write a CHUNK of samples into the circular buffers."""
    global write_idx, ring_filled
    n = ai_block.shape[1]
    assert enc_block.shape[0] == n

    end = write_idx + n
    if end <= BUF_LEN:
        ai_ring[:, write_idx:end] = ai_block
        enc_ring[write_idx:end] = enc_block
    else:
        k = BUF_LEN - write_idx
        ai_ring[:, write_idx:] = ai_block[:, :k]
        ai_ring[:, :n - k] = ai_block[:, k:]
        enc_ring[write_idx:] = enc_block[:k]
        enc_ring[:n - k] = enc_block[k:]
        ring_filled = True

    write_idx = (write_idx + n) % BUF_LEN
    if write_idx == 0:
        ring_filled = True

def get_latest_snapshot():
    """
    Returns (ai, enc) with time-ordered data from the circular buffers.
    If buffers haven't wrapped yet, returns the portion filled so far.
    """
    if not ring_filled:
        return ai_ring[:, :write_idx].copy(), enc_ring[:write_idx].copy()
    # Reorder so index 0 is the oldest sample
    ai_out = np.hstack((ai_ring[:, write_idx:], ai_ring[:, :write_idx]))
    enc_out = np.hstack((enc_ring[write_idx:], enc_ring[:write_idx]))
    return ai_out, enc_out

def main():
    # Arm the slave (encoder) first, then start the AI (master) to issue StartTrigger+SampleClock
    task.start()
    ai_task.start()

    try:
        while True:
            # Read exactly CHUNK samples from each, keeping them aligned
            ai_reader.read_many_sample(_ai_chunk, number_of_samples_per_channel=CHUNK, timeout=10.0)
            ctr_reader.read_many_sample_double(_enc_chunk, number_of_samples_per_channel=CHUNK, timeout=10.0)

            _ring_write(_ai_chunk, _enc_chunk)
            logger.write(_ai_chunk, _enc_chunk)
            # No prints, no plots. Your data is in ai_ring / enc_ring (circular).
            # Optional: do lightweight work here if needed.

    except KeyboardInterrupt:
        pass
    finally:
        # Stop & close cleanly
        for t in (task, ai_task):
            try: t.stop()
            except Exception: pass
            try: t.close()
            except Exception: pass
        logger.close()
        out_mat = logger.to_mat()
        print(out_mat)

if __name__ == "__main__":
    main()
    # Example (not executed until you stop with Ctrl+C):
    # ai_buf, enc_buf = get_latest_snapshot()
    # Now ai_buf.shape == (7, <= BUF_LEN) and enc_buf.shape == (<= BUF_LEN,)