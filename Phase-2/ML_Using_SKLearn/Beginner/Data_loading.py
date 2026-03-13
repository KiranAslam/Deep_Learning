import pandas as pd
import os
import time
from datetime import datetime

path = os.getcwd()

def key_stats(gather=" Total"):
    statpath = path + "/Data/Key_Stats.csv"
    stock_list = [x[0] for x in os.walk(path + "/Data/Key_Stats/")]
    print(stock_list)

key_stats()