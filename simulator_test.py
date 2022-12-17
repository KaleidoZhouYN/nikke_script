
from utils.win import get_hWnd_by_name
from config import config

from utils.simulator.base import Simulator
from utils.simulator.MuMuX import MuMuX
from utils.simulator.leidian import Leidian


s_map =  {r'MuMu':MuMuX,r'雷电':Leidian,'other':Simulator}
if __name__ == '__main__':
    simulator_name = r'MuMu'
    simulator = s_map[simulator_name]
    hWnd = get_hWnd_by_name(simulator_name)
    sim = simulator(hWnd,config)

    sim.start()
