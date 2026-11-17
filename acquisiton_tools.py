import time
import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType, Edge, TriggerType, EncoderType, AngleUnits, EncoderZIndexPhase, Level
from nidaqmx.stream_readers import AnalogMultiChannelReader, CounterReader
from nidaqmx.types import CtrFreq
from nidaqmx import constants
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import sys
from pymodbus.client import ModbusTcpClient
from datetime import datetime
import os
import scipy.io as io


class SyncedEncoderDAQ:
    def __init__(self, device="Dev4", counter="ctr0", ppr=1000, ai_rate=20000, ai_first_ch=0, ai_last_ch=5, buffer_secs=10, chunk=2000, encoder_enabled=True):
        """DAQ helper. If encoder_enabled is False the class will only perform AI acquisition.

        When encoder_enabled is True the encoder counter is configured and slave-synced to the AI sample clock.
        """
        self.DEVICE = device
        self.COUNTER = counter
        self.PPR = ppr
        self.AI_RATE = ai_rate
        self.AI_FIRST_CH = ai_first_ch
        self.AI_LAST_CH = ai_last_ch
        self.NUM_AI_CH = self.AI_LAST_CH - self.AI_FIRST_CH + 1
        self.CHUNK = chunk
        self.BUFFER_SECS = buffer_secs
        self.BUF_LEN = int(self.AI_RATE * self.BUFFER_SECS)

        self.has_encoder = bool(encoder_enabled)

        self.ai_ring = np.empty((self.NUM_AI_CH, self.BUF_LEN), dtype=np.float64)
        self.enc_ring = np.empty(self.BUF_LEN, dtype=np.float64)  # will be zeros if encoder disabled
        self.write_idx = 0
        self.ring_filled = False

        # Encoder/counter task (optional)
        self.ctr_task = None
        self.ctr_reader = None
        if self.has_encoder:
            self.ctr_task = nidaqmx.Task()
            enc_ch = self.ctr_task.ci_channels.add_ci_ang_encoder_chan(
                counter=f"{self.DEVICE}/{self.COUNTER}",
                decoding_type=EncoderType.X_2,
                zidx_enable=True,
                zidx_val=0.0,
                units=AngleUnits.DEGREES,
                zidx_phase=EncoderZIndexPhase.AHIGH_BHIGH,
                pulses_per_rev=self.PPR
            )
            enc_ch.ci_encoder_a_input_term = f"/{self.DEVICE}/PFI8"
            enc_ch.ci_encoder_b_input_term = f"/{self.DEVICE}/PFI10"
            enc_ch.ci_encoder_z_input_term = f"/{self.DEVICE}/PFI9"
            # enc_ch.ci_encoder_a_input_dig_fltr_enable = True
            # enc_ch.ci_encoder_b_input_dig_fltr_enable = True
            # enc_ch.ci_encoder_z_input_dig_fltr_enable = True
            # enc_ch.ci_encoder_a_input_dig_fltr_min_pulse_width = 6.425000e-6
            # enc_ch.ci_encoder_b_input_dig_fltr_min_pulse_width = 6.425000e-6
            # enc_ch.ci_encoder_z_input_dig_fltr_min_pulse_width = 6.425000e-6

        # AI task (always present)
        self.ai_task = nidaqmx.Task()
        self.ai_task.ai_channels.add_ai_voltage_chan(f"{self.DEVICE}/ai{self.AI_FIRST_CH}:{self.AI_LAST_CH}", min_val=-10.0, max_val=10.0)
        self.ai_task.timing.cfg_samp_clk_timing(
            rate=self.AI_RATE,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=self.BUF_LEN
        )

        # If encoder enabled, configure counter timing to use AI sample clock as source
        if self.has_encoder and self.ctr_task is not None:
            self.ctr_task.timing.cfg_samp_clk_timing(
                rate=self.AI_RATE,
                source=f"/{self.DEVICE}/ai/SampleClock",
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.BUF_LEN
            )
            arm = self.ctr_task.triggers.arm_start_trigger
            arm.trig_type = TriggerType.DIGITAL_EDGE
            arm.dig_edge_src = f"/{self.DEVICE}/ai/StartTrigger"
            arm.dig_edge_edge = Edge.RISING

        self.ai_reader = AnalogMultiChannelReader(self.ai_task.in_stream)
        if self.has_encoder and self.ctr_task is not None:
            self.ctr_reader = CounterReader(self.ctr_task.in_stream)

        self._ai_chunk = np.empty((self.NUM_AI_CH, self.CHUNK), dtype=np.float64)
        self._enc_chunk = np.empty(self.CHUNK, dtype=np.float64)

    def _ring_write(self, ai_block: np.ndarray, enc_block: np.ndarray = None):
        n = ai_block.shape[1]
        if enc_block is None:
            enc_block = np.zeros(n, dtype=np.float64)
        end = self.write_idx + n
        if end <= self.BUF_LEN:
            self.ai_ring[:, self.write_idx:end] = ai_block
            self.enc_ring[self.write_idx:end] = enc_block
        else:
            k = self.BUF_LEN - self.write_idx
            self.ai_ring[:, self.write_idx:] = ai_block[:, :k]
            self.ai_ring[:, :n - k] = ai_block[:, k:]
            self.enc_ring[self.write_idx:] = enc_block[:k]
            self.enc_ring[:n - k] = enc_block[k:]
            self.ring_filled = True
        self.write_idx = (self.write_idx + n) % self.BUF_LEN
        if self.write_idx == 0:
            self.ring_filled = True

    def get_latest_snapshot(self):
        if not self.ring_filled:
            return self.ai_ring[:, :self.write_idx].copy(), self.enc_ring[:self.write_idx].copy()
        ai_out = np.hstack((self.ai_ring[:, self.write_idx:], self.ai_ring[:, :self.write_idx]))
        enc_out = np.hstack((self.enc_ring[self.write_idx:], self.enc_ring[:self.write_idx]))
        return ai_out, enc_out

    def start(self):
        # Start counter/encoder first (if present) then AI
        if self.has_encoder and self.ctr_task is not None:
            self.ctr_task.start()
        self.ai_task.start()

    def stop(self):
        tasks = [self.ai_task]
        if self.has_encoder and self.ctr_task is not None:
            tasks.insert(0, self.ctr_task)
        for t in tasks:
            try:
                t.stop()
            except Exception:
                pass
            try:
                t.close()
            except Exception:
                pass


def read_ai_bias(duration=10.0, device="Dev4", ai_first_ch=0, ai_last_ch=5, rate=1000.0):
    """Read raw AI channels for `duration` seconds (not synced to encoder).

    Returns:
        data: ndarray shape (num_ch, N)
        times: ndarray shape (N,) (seconds, relative)

    This creates a temporary NI task, performs a finite acquisition, and closes the task.
    """
    num_ch = ai_last_ch - ai_first_ch + 1
    N = int(float(rate) * float(duration))

    task = nidaqmx.Task()
    try:
        task.ai_channels.add_ai_voltage_chan(f"{device}/ai{ai_first_ch}:{ai_last_ch}")
        task.timing.cfg_samp_clk_timing(rate=rate, sample_mode=AcquisitionType.FINITE, samps_per_chan=N)
        reader = AnalogMultiChannelReader(task.in_stream)
        buf = np.empty((num_ch, N), dtype=np.float64)
        task.start()
        # read_many_sample will block until samples acquired or timeout
        reader.read_many_sample(buf, number_of_samples_per_channel=N, timeout=max(10.0, duration + 5.0))
    finally:
        try:
            task.stop()
        except Exception:
            pass
        try:
            task.close()
        except Exception:
            pass

    times = np.arange(N, dtype=np.float64) / float(rate)

    # Convert bias voltages to forces using calibration matrix
    try:
        cal = cal_matrix()
        C = np.asarray(cal.cal_matrix)
        expected_ch = C.shape[1]
        if buf.shape[0] < expected_ch:
            raise ValueError(f"Not enough AI channels for calibration: need {expected_ch}, got {buf.shape[0]}")
        if buf.shape[0] > expected_ch:
            # Common situation: DAQ records an extra channel (e.g., 7 AI channels while FT sensor uses 6)
            print(f"Warning: read_ai_bias got {buf.shape[0]} channels but calibration expects {expected_ch}; using first {expected_ch} channels for force conversion.")
        ai_for_cal = buf[:expected_ch, :]
        # forces shape: (n_out, N)
        forces = C @ ai_for_cal
    except Exception as e:
        # If conversion fails, return raw voltages in place of forces to avoid breaking callers
        print(f"Failed to convert bias voltages to forces: {e}")
        forces = buf

    return forces, times

# This module contains the acquisition tools for an NI USB-6212 DAQ device for force/torque sensing, encoder reading and ESC control.
# Efe Sen, 12/02/2025


class wind_tunnel():
    def __init__(self, ip = '192.168.0.10', port = 502):
        self.ip = ip
        self.port = port
        self.client = ModbusTcpClient(ip, port = port)
        self.fit = [2.0882, -4.1089]

    def set_fan_speed(self, v):
        if v > 30:
            raise ValueError('Do not set the fan speed to > 30%.')
        
        write_address = 0
        write_data = int(v * 10)
        self.client.write_register(write_address, write_data)

    
    def set_U0(self, u):
        if u == 0:
            v = 0
        else:
            v = round(self.fit[0] * u + self.fit[1], 1)
        
        write_address = 0
        write_data = int(v * 10)
        
        if v > 96:
            raise ValueError('Do not set the fan speed to > 96%.')
        else:
            self.client.write_register(write_address, write_data)
    
    def stop(self):
        write_address = 0
        write_data = 0
        self.client.write_register(write_address, write_data)

    def read_PLC(self):
        N = 10
        T0 = np.zeros((N, 1))
        P0 = np.zeros((N, 1))

        for i in range(N):
            data = self.client.read_input_registers(address=0, count=16)
            T0[i] = data.registers[8] / 10
            P0[i] = data.registers[4] / 10
        T_mean = np.mean(T0)
        P_mean = np.mean(P0)

        return T_mean, P_mean

    def close(self):
        self.client.close()


class analog_mano_reader():
    def __init__(self, fs_rate=4000, buffer_size=4000):
        self.task = nidaqmx.Task()
        self.task.ai_channels.add_ai_voltage_chan("Dev2/ai0:1", min_val = -5.0 , max_val = 5.0)
        self.task.timing.cfg_samp_clk_timing(rate=fs_rate, sample_mode=constants.AcquisitionType.CONTINUOUS,
                                        samps_per_chan=buffer_size)
        self.stream = AnalogMultiChannelReader(self.task.in_stream)
    
    def start(self):
        self.task.start()

    def mano2pressure(self, raw):
        # raw is a 2x1 numpy array
        # Conversion from voltage to pressure in Pascals
        return 200 * raw

    def stop(self):
        self.task.stop()
        self.task.close()

class SecondaryAIRecorder:
    """Sample a small set of AI channels from another DAQ at low rate in a background thread.

    Usage:
        rec = SecondaryAIRecorder(device='Dev2', channels=[0,1], rate=10.0)
        rec.start()
        ...
        rec.stop()
        times, vals = rec.get_samples()  # vals shape (n_ch, N)
    """
    def __init__(self, device='Dev2', channels=[0,1], rate=10.0, buffer_len=10000):
        self.device = device
        self.channels = list(channels)
        self.rate = float(rate)
        self.interval = 1.0 / float(rate)
        self._running = False
        self._thread = None
        import threading
        self._lock = threading.Lock()
        self.buffer_len = int(buffer_len)

        # preallocate numpy ring buffers
        self._n_ch = len(self.channels)
        self._times = np.empty(self.buffer_len, dtype=np.float64)
        self._vals = np.empty((self._n_ch, self.buffer_len), dtype=np.float64)
        self._write_idx = 0
        self._count = 0

        # prepare NI task but don't start it yet
        self.task = nidaqmx.Task()
        ch_spec = ','.join([f"{self.device}/ai{ch}" for ch in self.channels])
        self.task.ai_channels.add_ai_voltage_chan(ch_spec, min_val=-5.0, max_val=5.0)

    def _run_loop(self):
        import time as _time
        reader = AnalogMultiChannelReader(self.task.in_stream)
        n_ch = self._n_ch
        while self._running:
            try:
                buf = np.empty((n_ch, 1), dtype=np.float64)
                reader.read_many_sample(buf, number_of_samples_per_channel=1, timeout=2.0)
                sample = buf[:, 0].copy()
            except Exception:
                # on read error, use NaNs
                sample = np.full((n_ch,), np.nan, dtype=np.float64)
            t = _time.time()
            with self._lock:
                self._times[self._write_idx] = t
                self._vals[:, self._write_idx] = sample
                self._write_idx = (self._write_idx + 1) % self.buffer_len
                if self._count < self.buffer_len:
                    self._count += 1
            _time.sleep(self.interval)

    def start(self):
        if self._running:
            return
        self.task.start()
        self._running = True
        import threading
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self.task.stop()
        except Exception:
            pass
        try:
            self.task.close()
        except Exception:
            pass

    def get_samples(self):
        with self._lock:
            if self._count == 0:
                return np.array([]), np.empty((self._n_ch, 0), dtype=np.float64)
            # compute start index of oldest sample
            start = (self._write_idx - self._count) % self.buffer_len
            if start + self._count <= self.buffer_len:
                times = self._times[start:start + self._count].copy()
                vals = self._vals[:, start:start + self._count].copy()
            else:
                # wrapped
                first_len = self.buffer_len - start
                times = np.concatenate((self._times[start:].copy(), self._times[:self._count - first_len].copy()))
                vals = np.hstack((self._vals[:, start:].copy(), self._vals[:, :self._count - first_len].copy()))
        return times, vals

class esc():
    """
    A class to control an ESC using NI-DAQmx.
    Methods
    -------
    __init__():
        Initializes the ESC task with a pulse channel and continuous timing.
    start():
        Starts the ESC task.
    stop():
        Stops the ESC task.
    write_pwm_ms(pwm_ms):
        Writes a PWM signal with the specified pulse width in milliseconds.
    write_throttle(throttle):
        Converts a throttle value to a PWM signal and writes it (0, 1)
    """
    def __init__(self):
        self.task = nidaqmx.Task()
        self.task.co_channels.add_co_pulse_chan_freq(
            "Dev2/ctr0", idle_state=Level.LOW, initial_delay=0.0,  # By default this is PFI12, which corresponds to pin 38 (P.2.4) on NI-USB-6212
            freq=50.0, duty_cycle=0.05  # Start at 0°
        )
        self.task.timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS)
        self.current_rpm = 0
        time.sleep(1)
        self.start()

    def start(self):
        self.task.start() 
        self.write_pwm_ms(1000) # Start at 0°

    def stop(self):
        self.task.stop() 
        self.task.close()

    def write_pwm_ms(self, pwm_ms): 
        try:
            if pwm_ms >= 1000 and pwm_ms <= 2000:
                duty = pwm_ms / 20000 # 20 ms period
                self.task.write(CtrFreq(50, duty)) # 50 Hz frequency
            else:
                print("PWM value must be between 1000 and 2000 us.")
        except:
            pass

    def write_throttle(self, throttle):
        if throttle >= 0 and throttle <= 1:
            pwm = 1060 + throttle * 1000 # normalized (0, 1) to (1000, 2000) us
            pwm = max(1000, min(pwm, 2000))
            self.write_pwm_ms(pwm)
        else:
            print("Throttle value must be between 0 and 1.")

    def write_rpm(self, rpm):
        if 1000 <= rpm <= 10000:
            pwm = 1004 + (rpm - 1000) * (2004 - 1004) / (10000 - 1000)
            self.write_pwm_ms(pwm)
            self.current_rpm = rpm
        elif rpm < 1000:
            self.write_pwm_ms(1004)
        else:
            print("RPM value must be between 1000 and 10000.")
    
    def ramp_to_rpm(self, target_rpm, step=100, delay=0.1):
        while self.current_rpm != target_rpm:
            if abs(self.current_rpm - target_rpm) < step:
                self.current_rpm = target_rpm
            elif self.current_rpm < target_rpm:
                self.current_rpm += step
                if self.current_rpm > target_rpm:
                    self.current_rpm = target_rpm
            else:
                self.current_rpm -= step
                if self.current_rpm < target_rpm:
                    self.current_rpm = target_rpm
            self.write_rpm(self.current_rpm)
            time.sleep(delay)


def ask_user(in_string):
    global running
    input(in_string)
    print("\n")
    running = False

def J2RPM(J, V, D):
    n = V / (J*D)
    RPM = n * 60
    return RPM

def RPM2J(RPM, V, D):
    n = RPM / 60
    J = V / (n * D)
    return J 

def J2U(J, RPM, D):
    n = RPM / 60
    V = J * n * D
    return V

def air_density(T, P, humidity):
    """
    Calculate the air density using temperature, pressure, and humidity.
    Parameters:
    T (float): Temperature in Celsius.
    P (float): Pressure in Pascals.
    humidity (float): Relative humidity in percentage (0-100).
    Returns:
    float: Air density in kg/m^3.
    """
    # Constants
    R_specific = 287.05  # Specific gas constant for dry air in J/(kg·K)
    R_vapor = 461.495  # Specific gas constant for water vapor in J/(kg·K)
    T_kelvin = T + 273.15  # Convert temperature to Kelvin

    # Saturation vapor pressure (in Pascals) using Tetens formula
    e_s = 6.112 * np.exp((17.67 * T) / (T + 243.5)) * 100

    # Actual vapor pressure (in Pascals)
    e = humidity / 100 * e_s

    # Partial pressure of dry air (in Pascals)
    P_dry = P - e

    # Air density calculation
    density = (P_dry / (R_specific * T_kelvin)) + (e / (R_vapor * T_kelvin))
    return density

class cal_matrix():
    def __init__(self):
            # self.cal_matrix = np.array([
            # [0.00371,  -0.02167,   0.00473,   3.15117,   0.02060,  -3.17213], # ATI Mini 40 IP66 calibration matrix SOTON BLWT, 12/02/2025
            # [0.00433,  -3.79734,  -0.01068,   1.75137,  -0.00602,   1.87770], 
            # [5.14105,  -0.08149,   5.23352,  -0.13159,   5.39405,  -0.12217],
            # [-0.00134,  -0.04024,   0.07490,   0.01583,  -0.07733,   0.02243], 
            # [-0.08373,   0.00302,   0.04110,  -0.03512,   0.04666,   0.03224],
            # [0.00106,  -0.04752,   0.00002,  -0.04370,  -0.00002,  -0.04564]])
            self.cal_matrix = np.array([
            [0.00620,   0.07157,   0.84050,  12.70886,  -0.52398, -11.83592], # ATI Mini 40, FT47790 calibration matrix SOTON BLWT, 23/03/2025
            [-0.41651, -14.70327, 0.63536, 7.34283, 0.26233, 6.89303], 
            [20.82344, -0.06424, 20.90322, 0.60811, 21.03796, -0.02464],
            [-0.00570, -0.08270, 0.29727, 0.04899, -0.29664, 0.03775], 
            [-0.33797, 0.00221, 0.16560, -0.06711, 0.17786, 0.06643],
            [-0.00465, -0.17603, -0.00794, -0.17881, -0.00924, -0.16388]])  
            
    def voltage2force(self, raw):
        # raw is a 6x1 numpy array
        # Conversion from voltage to force in Newtons
        return self.cal_matrix @ raw


def ai_voltages_to_forces(ai_array=None, h5_path=None, cal=None):
    """Convert AI voltages to forces using calibration matrix.

    Provide either `ai_array` (ndarray shape (num_ch, N)) or `h5_path` pointing
    to an HDF5 file produced by `H5MatLogger` (function will read dataset 'ai'
    and 'time').

    Returns a tuple (forces, times) where `forces` is ndarray (n_out, N) and
    `times` is ndarray (N,) when available (or None if not).
    """
    # If given an h5 path, read ai and time from it
    times = None
    if h5_path is not None:
        try:
            import h5py
        except Exception as e:
            raise RuntimeError("h5py is required to read HDF5 files") from e
        with h5py.File(h5_path, 'r') as f:
            ai_array = f['ai'][...]
            if 'time' in f:
                times = f['time'][...]

    if ai_array is None:
        raise ValueError('Either ai_array or h5_path must be provided')

    arr = np.asarray(ai_array)
    if arr.ndim != 2:
        raise ValueError("ai_array must be 2D (num_ch, N)")
    if cal is None:
        cal = cal_matrix()
    C = np.asarray(cal.cal_matrix)
    # C shape should be (n_out, num_ch)
    if C.shape[1] != arr.shape[0]:
        raise ValueError(f"Calibration matrix expects {C.shape[1]} channels but ai has {arr.shape[0]}")
    forces = C @ arr
    return forces, times

    
def get_latest_file():
    folder_path = "bias"
    
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    latest_file = max(files, key=os.path.getctime)
    return latest_file
