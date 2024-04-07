# License: MIT

# import sys
# import os
# import numpy as np
# import multiprocessing
import subprocess
import shlex
import time
import threading
import redis
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from openbox import Optimizer, space as sp

from env import carla_manage


# Define Search Space
space = sp.Space()
x1 = sp.Real("x1", 0, 1, default_value=0.0)
x2 = sp.Real("x2", 0, 1, default_value=0.0)
space.add_variables([x1, x2])


def loop_manage(FAULT_PARAMETER_1, FAULT_PARAMETER_2):
    """
    用于根据 FAULT_PARAMETER_1, FAULT_PARAMETER_2 两个参数控制 START_INJECT 
    故障注入标识符
    FAULT_PARAMETER_1: 优化参数1
    FAULT_PARAMETER_2: 优化参数2
    """
    
    while r.get('STOP_EXPERIMENT') == '0':
        # 故障注入时间大于 0 才注入故障
        if float(FAULT_PARAMETER_1) > 0.0:
            r.set('START_INJECT', 1)
        time.sleep(float(FAULT_PARAMETER_1))
        r.set('START_INJECT', 0)
        time.sleep(float(FAULT_PARAMETER_2))
    

def manage_experiment_time(delay_time):
    """
    用于控制故障注入标识符的线程的回调函数 传入参数为实验开始到故障注入的延迟时间
    delay_time: 实验开始时第一次故障延时注入时间
    """
    # 开始实验时间
    START_EXPERIMENT_TIME = 0
    # 开始注入故障数据时间
    START_INJECT_TIME = 0
    # 获取当前系统时间
    current_time = datetime.now()
    # 获取故障注入持续时常
    FAULT_PARAMETER_1 = r.get('FAULT_PARAMETER_1')
    # 获取故障间隔时常
    FAULT_PARAMETER_2 = r.get('FAULT_PARAMETER_2')
    # r.set('START_INJECT_TIME', 3)
    # 等待 scenario runner 中写入开始实验时间
    while r.get('START_EXPERIMENT_TIME') == '':
        START_EXPERIMENT_TIME = ''

    START_EXPERIMENT_TIME = datetime.fromisoformat(r.get('START_EXPERIMENT_TIME'))

    # 设置首次故障注入的时间
    START_INJECT_TIME = timedelta(seconds=delay_time) + START_EXPERIMENT_TIME
    r.set('START_INJECT_TIME', START_INJECT_TIME.isoformat())
    
    # 等待当前时间等于设定的故障注入时间时设置故障注入标识符
    # 故障注入标识符用于指导pylot是否使用故障数据
    while abs(current_time - START_INJECT_TIME) > timedelta(seconds=1):
        current_time = datetime.now()

    # 设置故障注入标识符
    r.set('START_INJECT', 1)

    # 创建线程并启动
    loop_thread = threading.Thread(target=loop_manage, args=(FAULT_PARAMETER_1, FAULT_PARAMETER_2))
    loop_thread.start()
    loop_thread.join()


# Define Objective Function
def branin(config):

    # 每轮实验数据重置
    r.mset({
        'START_INJECT': 0, 
        'START_EXPERIMENT_TIME': '', 
        'START_INJECT_TIME': '', 
        'FAULT_PARAMETER_1': 0, 
        'FAULT_PARAMETER_2': 0, 
        'STOP_EXPERIMENT': 0,
        'OPTIMISATION_FUNCTION_VALUE': 0
        })

    carla_man = carla_manage.CarlaManage()
    # 启动carla服务器
    carla_man.start_carla_server()

    x1, x2 = config['x1'], config['x2']
    r.set('FAULT_PARAMETER_1', x1)
    r.set('FAULT_PARAMETER_2', x2)

    # 此线程用于根据实验开始时间和故障开始注入时间来设置故障注入标识符
    try:
        t = threading.Thread(target=manage_experiment_time, args=(2,))
        t.start()
    except:
        print("Error: unable to start thread")

    # 调用 manage.py
    # 创建子进程,输出内容重定向
    log = open("logs/pylot_log" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ".txt", 'a')
    p = subprocess.Popen(shlex.split("python ./env/env_manage.py"), stderr = log, shell=False)
    p.wait()
    log.close()

    # 关闭carla服务器
    carla_man.stop_carla_server()
    
    # 返回目标函数值
    y = int(r.get('OPTIMISATION_FUNCTION_VALUE'))

    print("优化函数值 = ", -y)
    # 优化目标好像是最小化
    return {'objectives': [-y]}
    

# Run
if __name__ == "__main__":

    pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
    r = redis.Redis(connection_pool=pool)

    """
    redis 数据含义:
        START_INJECT:是否注入故障标识符 为 1 时注入故障数据
        START_EXPERIMENT_TIME:实验开始时间 
        FAULT_PARAMETER_1:注入故障持续时间
        FAULT_PARAMETER_2:注入故障间隔时间
        START_EXPERIMENT:一次实验结束标志
        OPTIMISATION_FUNCTION_VALUE:优化函数值
    """
    r.mset({
        'START_INJECT': 0, 
        'START_EXPERIMENT_TIME': '', 
        'START_INJECT_TIME': '', 
        'FAULT_PARAMETER_1': 0, 
        'FAULT_PARAMETER_2': 0, 
        'STOP_EXPERIMENT': 0,
        'OPTIMISATION_FUNCTION_VALUE': 0
        })

    opt = Optimizer(
        branin,
        space,
        max_runs=2,
        # surrogate_type='gp',
        surrogate_type='auto',
        task_id='quick_start',
        # Have a try on the new HTML visualization feature!
        # visualization='advanced',   # or 'basic'. For 'advanced', run 'pip install "openbox[extra]"' first
        # auto_open_html=True,        # open the visualization page in your browser automatically
    )
    history = opt.run()

    print(history)

    history.plot_convergence(true_minimum=0.397887)

    plt.show()

    # install pyrfr to use get_importance()
    # print(history.get_importance())

    # Have a try on the new HTML visualization feature!
    # You can also call visualize_html() after optimization.
    # For 'show_importance' and 'verify_surrogate', run 'pip install "openbox[extra]"' first
    # history.visualize_html(open_html=True, show_importance=True, verify_surrogate=True, optimizer=opt)
