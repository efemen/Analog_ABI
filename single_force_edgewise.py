from acquisiton_tools import SyncedEncoderDAQ, wind_tunnel, air_density, read_ai_bias, ai_voltages_to_forces, SecondaryAIRecorder, esc
from active_grid_tools import active_grid
from h5_mat_logger import H5MatLogger
from scipy import io
import os
import time
from datetime import datetime


def run_single_force(U_inf=6, D_inch=16, pitch=55, RPM=3250, station_time=120):
	# Setup
	turbulence_case = "D"
	run_path = f"results/{turbulence_case}/"
	wt_fan_speed = 13.8
	os.makedirs(run_path, exist_ok=True)
	phase_angle = 0
	save_path = os.path.join(run_path, f"U{U_inf}_D{D_inch}_{pitch}_RPM{RPM}.mat")

	# If the save_path already exists, append a run counter: file_run01.mat, run02, ...
	if os.path.exists(save_path):
		base, ext = os.path.splitext(save_path)
		cnt = 1
		new_path = f"{base}_run{cnt:02d}{ext}"
		while os.path.exists(new_path):
			cnt += 1
			new_path = f"{base}_run{cnt:02d}{ext}"
		save_path = new_path
		print(f"Save path exists; using new path: {save_path}")


	WT = wind_tunnel()
	ag = active_grid()
	ag.go_home()

	print("Reading WT temperature and pressure...")
	temperature, pressure = WT.read_PLC()
	# temperature, pressure = 22.0, 101.325  # Dummy values for testing without WT
	humidity = 0.71
	rho = air_density(temperature, pressure, humidity) * 100
	print("Target q = ", 0.5 * rho * U_inf**2)

	D_m = D_inch * 0.0254
	esc_control = esc()
	time.sleep(3)  # Give some time to connect

	# Start DAQ
	daq = SyncedEncoderDAQ()

	# Start secondary low-rate manometer recorder on Dev2 ai0/ai1 at 5 Hz
	man_rec = SecondaryAIRecorder(device='Dev2', channels=[0,1], rate=5.0)

	# Pre-run bias capture (10s) using a temporary AI read
	print("Capturing pre-run bias...")
	bias_pre, bias_pre_time = read_ai_bias(duration=10.0, device=daq.DEVICE, ai_first_ch=daq.AI_FIRST_CH, ai_last_ch=daq.AI_LAST_CH, rate=daq.AI_RATE)

	print('Ramping up wind tunnel...')
	

	if turbulence_case == "B":
		ag.run_case(1)  # Start active grid case B
		time.sleep(15)
	elif turbulence_case == "C":
		ag.run_case(2)  # Start active grid case C
		time.sleep(15)
	elif turbulence_case == "D":
		ag.run_case(3)  # Start active grid case D
		time.sleep(15)
	elif turbulence_case == "E":
		ag.run_case(4)  # Start active grid case E
		time.sleep(15)
	else:
		raise ValueError(f"Unknown turbulence_case: {turbulence_case}")
	
	WT.set_fan_speed(wt_fan_speed)  # Set to ~8 m/s
	time.sleep(20)  # Wait for flow to stabilize

	# Create logger to stream chunks to HDF5 (and optionally convert to .mat later)
	logger = H5MatLogger(
		path=None,
		num_ai_ch=daq.NUM_AI_CH,
		chunk_len=daq.CHUNK,
		ai_rate=daq.AI_RATE,
		device=daq.DEVICE,
		ai_channel_names=[f"ai{i}" for i in range(daq.AI_FIRST_CH, daq.AI_LAST_CH + 1)],
		ppr=daq.PPR,
		notes=f"U_inf={U_inf}, D_inch={D_inch}, pitch={pitch}, RPM={RPM}",
	)

	# Attach additional metadata from wind tunnel read
	logger.write_attrs({
		"temperature": float(temperature),
		"pressure": float(pressure),
		"humidity": float(humidity),
		"rho": float(rho),
		"U_inf": float(U_inf),
		"D_inch": float(D_inch),
		"D_m": float(D_m),
		"pitch": float(pitch),
		"RPM": float(RPM),
		"turbulence_case": turbulence_case,
		"phase_angle": phase_angle,
	})

	print("Ramping to target RPM...")
	esc_control.ramp_to_rpm(RPM)
	time.sleep(5)  # Give some time to ramp up

	daq.start()
	man_rec.start()

	print(f"Starting run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	print(f"Recording for {station_time} seconds...")

	t0 = time.time()
	
    # MAIN LOOP: RUN THE EXPERIMENT HERE:

	try:
		while time.time() - t0 < station_time:
			# Read CHUNK samples and append to ring via internal readers
			daq.ai_reader.read_many_sample(daq._ai_chunk, number_of_samples_per_channel=daq.CHUNK, timeout=10.0)
			daq.ctr_reader.read_many_sample_double(daq._enc_chunk, number_of_samples_per_channel=daq.CHUNK, timeout=10.0)
			daq._ring_write(daq._ai_chunk, daq._enc_chunk)
			# Stream to HDF5
			logger.write(daq._ai_chunk, daq._enc_chunk)
			time.sleep(0.01)  # small sleep to avoid tight loop
			
	except KeyboardInterrupt:
		print("Interrupted by user.")
		
	finally:
		# Post-run bias capture
		esc_control.ramp_to_rpm(0)
		ag.stop_case()
		ag.go_home()
		man_times, man_vals = man_rec.get_samples()
		man_rec.stop()
		daq.stop()
		WT.set_fan_speed(0)
		# Wait a bit for flow to die down before capturing bias
		time.sleep(20)

		print(f"Ending run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
		print("Capturing post-run bias...")
		bias_post, bias_post_time = read_ai_bias(duration=10.0, device=daq.DEVICE, ai_first_ch=daq.AI_FIRST_CH, ai_last_ch=daq.AI_LAST_CH, rate=daq.AI_RATE)
		esc_control.stop()
		# Stop secondary recorder, close logger, and convert saved HDF5 to .mat

		logger.close()
		# Compute forces from saved AI voltages and include in extra_arrays
		force_N, time_saved = ai_voltages_to_forces(h5_path=logger.path)
		# Build extra arrays dict and save to .mat (force arrays added unconditionally)
		extra = {
			"bias_pre": bias_pre,
			"bias_pre_time": bias_pre_time,
			"bias_post": bias_post,
			"bias_post_time": bias_post_time,
			"force_N": force_N,
			"force_time": time_saved,
			"manometers": man_vals * 200, # Scale to Pa, 200 Pa/V
			"manometers_time": man_times,
			"temperature": float(temperature),
			"pressure": float(pressure),
			"humidity": float(humidity),
			"rho": float(rho),
			"U_inf": float(U_inf),
			"D_inch": float(D_inch),
			"D_m": float(D_m),
			"pitch": float(pitch),
			"RPM": float(RPM),
			"turbulence_case": turbulence_case,
			"phase_angle": phase_angle,
		}
		try:
			out_mat = logger.to_mat(save_path, extra_arrays=extra)
			print(f"Wrote MAT: {out_mat}")
		except Exception as e:
			print(f"Logger save failed: {e}")
		# Remove intermediate HDF5
		try:
			if os.path.exists(logger.path):
				os.remove(logger.path)
				print(f"Removed intermediate HDF5: {logger.path}")
		except Exception as e:
			print(f"Failed to remove HDF5 file: {e}")

if __name__ == '__main__':
	run_single_force()