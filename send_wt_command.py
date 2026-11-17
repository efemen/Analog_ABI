from acquisiton_tools import wind_tunnel

WT = wind_tunnel()


# WT.set_fan_speed(9.2) # ~4 m/s
WT.set_fan_speed(11.3) # ~4 m/s

# WT.set_fan_speed(16.4)
# WT.set_fan_speed(13.6)

# WT.set_fan_speed(19.5) # D
# WT.set_fan_speed(16.3) # A Passive grid
# WT.set_fan_speed(13.4) # ~8 m/s
WT.set_fan_speed(0)


# WT.set_fan_speed(23.5) # B C
# WT.set_fan_speed(19.5) # D E
WT.close()