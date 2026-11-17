import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType, Level
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.types import CtrFreq
from nidaqmx import constants
from nidaqmx.constants import AcquisitionType, EncoderType, AngleUnits, EncoderZIndexPhase
import matplotlib.pyplot as plt
import matplotlib as mpl

DEVICE = "Dev4"
COUNTER = "ctr0"
PPR = 1000

task = nidaqmx.Task()
enc_ch = task.ci_channels.add_ci_ang_encoder_chan(
    counter=f"{DEVICE}/{COUNTER}",
    decoding_type=EncoderType.X_2,
    zidx_enable=True,                     # reset at index
    zidx_val=0.0,                         # reset angle to 0.0 on Z
    units=AngleUnits.DEGREES,
    zidx_phase=EncoderZIndexPhase.AHIGH_BHIGH,  # adjust to match your Z polarity/phase
    pulses_per_rev=PPR
)

enc_ch.ci_encoder_a_input_term = f"/{DEVICE}/PFI8"
enc_ch.ci_encoder_z_input_term = f"/{DEVICE}/PFI9"
enc_ch.ci_encoder_b_input_term = f"/{DEVICE}/PFI10"

# enc_ch.ci_encoder_a_input_dig_fltr_enable = True
# enc_ch.ci_encoder_b_input_dig_fltr_enable = True
# enc_ch.ci_encoder_z_input_dig_fltr_enable = True

# enc_ch.ci_encoder_a_input_dig_fltr_min_pulse_width = 2e-6
# enc_ch.ci_encoder_b_input_dig_fltr_min_pulse_width = 2e-6
# enc_ch.ci_encoder_z_input_dig_fltr_min_pulse_width = 2e-6

def main():
    # Start the encoder task (on-demand reads)
    task.start()

    # --- Matplotlib polar setup ---
    plt.ion()
    fig = plt.figure("Encoder Angle (Polar)", figsize=(8,8))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_ylim(0, 1.0)                         # radius from 0 to 1
    ax.set_yticklabels([])                      # hide radial ticks
    ax.set_theta_direction(-1)                  # optional: clockwise
    ax.set_theta_zero_location("N")             # optional: 0° at top

    # Radar-style colors (blue variant)
    fig.patch.set_facecolor("#000010")        # near-black figure background with blue tint
    ax.set_facecolor("#001022")               # very dark blue radial background
    mpl.rcParams.update({"text.color": "#66CCFF"})

    # Grid / concentric rings in dark blue
    ax.grid(True, color="#002244", linestyle='-', linewidth=0.6, alpha=0.7)
    # draw a few concentric circles to resemble radar ranges
    theta_full = np.linspace(0, 2*np.pi, 360)
    for r in (0.25, 0.5, 0.75, 1.0):
        ax.plot(theta_full, np.full_like(theta_full, r), color="#002244", linewidth=0.8, alpha=0.6)

    # Draw 7 evenly spaced radial lines (spokes) to match your 14 pole-pair motor snap positions
    spokes = np.linspace(0, 2*np.pi, 14, endpoint=False)
    for a in spokes:
        ax.plot([a, a], [0, 1], color="#002244", linewidth=1.0, alpha=0.85)
    # align theta ticks to the spokes but hide labels for a clean radar look
    ax.set_xticks(spokes)
    ax.set_xticklabels([])

    # A bright blue sweep line (radius 0→1) and a glowing dot at the tip
    line, = ax.plot([0, 0], [0, 1], linewidth=3, color="#66CCFF", solid_capstyle='round')
    dot,  = ax.plot([0], [1], marker="o", markersize=8, color="#66CCFF", markeredgecolor="#001022", markeredgewidth=0.8)


    try:
        while plt.fignum_exists(fig.number):
            # Read a single on-demand sample (degrees)
            angle_deg = task.read()
            # Protect against any wrap formatting from DAQmx (keep 0..360)
            angle_deg = float(angle_deg) % 360.0
            theta = np.deg2rad(angle_deg)

            # Update the polar line and point
            line.set_data([theta, theta], [0, 1])
            dot.set_data([theta], [1])


            plt.pause(0.001)  # ~1000 Hz UI refresh

    except KeyboardInterrupt:
        pass
    finally:
        # Clean shutdown
        try:
            task.stop()
        except Exception:
            pass
        task.close()
        plt.ioff()
        plt.close(fig)

if __name__ == "__main__":
    main()
