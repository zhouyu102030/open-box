import subprocess
import os
import time
# import redis
import logging
from time import sleep


class PylotManager:
    """
    管理 Pylot 的类，用于启动和关闭 Pylot
    """
    def __init__(self):
        self.exec_command = [
            "docker", "exec", "b34be4a13a5a", "bash", "-c",
            "PYTHONPATH=~/workspace/pylot/dependencies/CARLA_0.9.10.1/\
                PythonAPI/carla:~/workspace/pylot/dependencies:\
            $PYTHONPATH python pylot.py --flagfile=configs/scenarios/\
                person_avoidance_frenet.conf --simulator_host=172.17.0.1"
        ]
        self.restart_command = ["docker", "restart", "b34be4a13a5a"]
        self.start_command = ["docker", "start", "b34be4a13a5a"]
        self.stop_command = ["docker", "stop", "b34be4a13a5a"]


    def run(self):
        try:
            # 等待 SR 准备完成
            time.sleep(3)
            subprocess.Popen(self.start_command, shell=False)

            # 等待容器启动
            sleep(2)

            pid = os.fork()
            # 子进程执行 此进程可能会随着容器的关闭而关闭，从而不需要考虑资源回收
            if pid == 0:
                os.execvp(self.exec_command[0], self.exec_command)
                
        except Exception as e:
            print("pylot子进程创建失败:", e)


    def stop(self):
        try:
            subprocess.Popen(self.stop_command, shell=False)
        except Exception as e:
            print("pylot容器关闭失败:", e)


class ScenarioManage:
    """
    管理 ScenarioRunner 的类，用于启动和关闭 ScenarioRunner
    """
    def __init__(self):
        # self.sr_command = [
        #     "/home/dell/miniconda3/envs/scenario/bin/python",
        #     "/home/dell/Soft/scenario_runner-0.9.13/scenario_runner.py",
        #     "--scenario", "FollowLeadingVehicle_1", "--reloadWorld", "--file"
        # ]
        self.sr_command = [
            "/home/dell/miniconda3/envs/scenario/bin/python",
            "/home/dell/Soft/scenario_runner-0.9.13/scenario_runner.py",
            "--scenario", "FollowLeadingVehicle_1", "--reloadWorld"
        ]
        self.srProcess = None


    def run(self):
        try:
            log = open("scenario.txt", 'a')
            self.srProcess = subprocess.Popen(self.sr_command, stdout = log, stderr = log, shell=False)
        except subprocess.CalledProcessError as e:
            print("SR子进程创建失败:", e)


    def stop(self):
        sleep(1)
        self.srProcess.communicate()


if __name__ == "__main__":
    """
    注意：
        1、pylot撞击停止的前车问题可以通过让前车不停规避
            解决思路：
                重新设计实验，让前车不停
        2、优化目标函数还没定义
            实现思路:
                明确优化目标
                现有文献大概是以加速找出引起碰撞的故障参数为优化目标的
                这里还需要多看文献和思考
        3、故障参数和优化目标函数结果传递还未实现
            实现思路：
                可以使用redis进行数据交互
        4、Carla服务器经常崩溃问题没有解决(已解决)
            解决思路:
                启动carla后 通过查询2000端口号的pid号来检查carla的运行状态
                并由carlamanag/redie维护更系carla的pid号和运行状态
                优化算法在执行优化之前需要检查carla的运行状态
                若carla服务器未运行则需要重启carla(可能要注意carla关闭的时刻)

                或者直接在一次实验结束后关闭carla(比较简单的做法 已实现)

                pylot的启动和关闭方式使用容器内进程管理 而不是粗暴重启容器(已验证不能解决问题)
    """

    if not os.path.exists("./tmp_pipe"):
        os.mkfifo("./tmp_pipe")

    srManager = ScenarioManage()
    srManager.run()

    # 等待场景准备完成
    sleep(5)

    # 重启 pylot 容器，容器中运行 pylot
    pyManager = PylotManager()
    pyManager.run()

    # 阻塞等待管道写入信息，信息为SR结束即将结束运行的信号
    fifo_fd = os.open("./tmp_pipe", os.O_RDONLY)
    msg = os.read(fifo_fd, 1024)
    os.close(fifo_fd)

    # 接收到SR结束的信号
    if fifo_fd:
        # 结束pylot
        srManager.stop()

    pyManager.stop()
    # carlaManage.stop_carla_server()

    # ToDo：返回优化结果