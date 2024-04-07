from datetime import datetime
from time import sleep
import redis

def display_key_values(keys):
    # 连接到本地Redis服务器
    pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
    r = redis.Redis(connection_pool=pool)

    while True:
        print(
            keys[0], ' = ', r.get(keys[0]),'\n',
            keys[1], ' = ', r.get(keys[1]),'\n',
            keys[2], ' = ', r.get(keys[2]),'\n',
            keys[3], ' = ', r.get(keys[3]),'\n',
            keys[4], ' = ', r.get(keys[4]),'\n',
            keys[5], ' = ', r.get(keys[5]),'\n',
            keys[6], ' = ', r.get(keys[6]),'\n',
            "-------------------------------"
            )
        sleep(1)
        

if __name__ == "__main__":
    keys = ['START_INJECT', 'START_EXPERIMENT_TIME', 'START_INJECT_TIME', 'FAULT_PARAMETER_1', 
            'FAULT_PARAMETER_2', 'STOP_EXPERIMENT', 'OPTIMISATION_FUNCTION_VALUE']
    display_key_values(keys)
