from active_grid_tools import active_grid  
import time

ag = active_grid()

try:
    ag.stop_case()
except:
    pass

ag.stop()

ag.go_home()