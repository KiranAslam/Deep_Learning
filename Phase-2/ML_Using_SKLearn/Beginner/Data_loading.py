import pandas as pd
import os
import time
from datetime import datetime

path = os.getcwd()

def key_stats(gather=" Total"):
    statpath = path + "/Data/Key_Stats.csv"
    stock_list = [x[0] for x in os.walk(path + "/Data/Key_Stats/")]
    print(stock_list)

    for each in stock_list[1:]:
        each_file = os.listdir(each)
        if len(each_file) > 0:
            for file in each_file:
                date_stamp = datetime.strptime(file, "%Y%m%d%H%M%S.html")
                unix_time = time.mktime(date_stamp.timetuple())
                print(date_stamp, unix_time)
                time.sleep(15)

key_stats()