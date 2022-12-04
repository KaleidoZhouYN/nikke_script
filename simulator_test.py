#from utils.strategy import MuMuX
from utils.simulator.MuMuX import MuMuX

from utils.win import get_hWnd_by_name
from config import config
from easydict import EasyDict as edict
config = edict(config)

if __name__ == '__main__':
    keyword = r'MuMu'
    hWnd = get_hWnd_by_name(keyword)
    sim = MuMuX(hWnd,config)

    sim.start_simulation()
