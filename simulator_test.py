
from utils.win import get_hWnd_by_name
from config import config

from utils.simulator.base import Simulator
from utils.simulator.MuMuX import MuMuX
from utils.simulator.leidian import Leidian

import argparse
def parser():
    parse = argparse.ArgumentParser()
    parse.add_argument('--simulator_name',type=str)
    parse.add_argument('--hWnd',type=int)
    args = parse.parse_args()

    return args

s_map =  {'MuMu':MuMuX,r'雷电':Leidian,'other':Simulator}
if __name__ == '__main__':
    args = parser()
    simulator = s_map[args.simulator_name]
    sim = simulator(args.hWnd,config)

    sim.start_simulation()
