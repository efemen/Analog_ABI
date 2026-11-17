from acquisiton_tools import esc
import time

def main():
	esc_control = esc()
	print("Enter integer RPM values between 1000 and 10000 to send commands to the ESC.")
	print("Type 'exit' to quit.")


	while True:
		cmd = input("Command: ").strip()
		if cmd.lower() == 'exit':
			break

		if not cmd.isdigit():
			print("Please enter a valid integer or 'exit' to quit.")
			continue

		value = int(cmd)
		if value < 0 or value > 3500:
			print("Value out of range (0 to 3500). Try again.")
			continue

		esc_control.ramp_to_rpm(value)
		time.sleep(0.1)  # Give ESC time to process

	esc_control.stop()
	print("ESC control stopped.")

if __name__ == "__main__":
	main()