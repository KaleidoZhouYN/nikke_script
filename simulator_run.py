
from utils.win import get_hWnd_by_name
from config import config

from utils.simulator.base import Simulator
from utils.simulator.MuMuX import MuMuX
from utils.simulator.leidian import Leidian

import threading
import logging

f_handler = logging.FileHandler(filename='./logs/main_process.txt',
                            mode = 'w',
                            encoding='utf-8')
logger = logging.getLogger("main_process")
logger.setLevel(logging.INFO)
logger.addHandler(f_handler)

import argparse
def parser():
    parse = argparse.ArgumentParser()
    parse.add_argument('--simulator_name',type=str)
    parse.add_argument('--hWnd',type=int)
    args = parse.parse_args()

    return args

# datas=[('C:\\Python310\\Lib\\site-packages\\onnxruntime\\capi\\onnxruntime_providers_shared.dll','onnxruntime\\capi')]
"""
def receive_stdin(self,logger):
    logger.info("监听线程已开启")
    while 1:
        msg = input()
        logger.info(msg)
        if msg == 'start_aim':
            self._start_aim = 1
        if msg == 'end_aim':
            self._end_aim = 1

        if msg == 'start_simroom':
            logger.info('get simroom start')
            self._start_sim = 1
        if msg == 'end_simroom':
            self._end_sim = 1
"""
s_map =  {'MuMu':MuMuX,r'雷电':Leidian,'other':Simulator}
if __name__ == '__main__':
    args = parser()
    simulator = s_map[args.simulator_name]
    sim = simulator(args.hWnd,config)

    sim.start()
