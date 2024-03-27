# License: MIT

import sys
import os
import numpy as np
import multiprocessing
import subprocess
import shlex
import time
import matplotlib.pyplot as plt
from openbox import Optimizer, space as sp

from env import carla_manage


# Define Search Space
space = sp.Space()
x1 = sp.Real("x1", -5, 10, default_value=0)
x2 = sp.Real("x2", 0, 15, default_value=0)
space.add_variables([x1, x2])


# Define Objective Function
def branin(config):

    carla_man = carla_manage.CarlaManage()
    # 启动carla服务器
    carla_man.start_carla_server()

    x1, x2 = config['x1'], config['x2']

    # # 创建 pipe 用于传递优化参数值
    # if not os.path.exists("./variant_pipe"):
    #     os.mkfifo("./variant_pipe")

    # TODO:将优化参数值传递给 manage.py

    # # 从 manage.py 中返回优化函数结果值
    # if not os.path.exists("./result_pipe"):
    #     os.mkfifo("./result_pipe")

    # 调用 manage.py
    # 创建子进程,输出内容重定向
    log = open("manage.txt", 'a')
    p = subprocess.Popen(shlex.split("python ./env/env_manage.py"), stdout = log, stderr = log, shell=False)
    p.wait()
    log.close()

    # 关闭carla服务器
    carla_man.stop_carla_server()
    
    # TODO:返回目标函数值

    y = 1
    return {'objectives': [y]}


# Run
if __name__ == "__main__":

    opt = Optimizer(
        branin,
        space,
        max_runs=10,
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

    # TODO: carla server stop

    plt.show()

    # install pyrfr to use get_importance()
    # print(history.get_importance())

    # Have a try on the new HTML visualization feature!
    # You can also call visualize_html() after optimization.
    # For 'show_importance' and 'verify_surrogate', run 'pip install "openbox[extra]"' first
    # history.visualize_html(open_html=True, show_importance=True, verify_surrogate=True, optimizer=opt)
