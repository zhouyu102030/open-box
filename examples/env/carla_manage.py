import subprocess
import logging
import time
from time import sleep
from datetime import datetime

# class CarlaMonitor:
#     def __init__(self, carla_process) -> None:
#         self.carla_state = 0
#         self.port = 2000
#         self.log = "carla_log" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ".txt"
#         self.carla_commend = ["bash", "/home/dell/Soft/CARLA_0.9.13/CarlaUE4.sh"]
#         self.carla_process = carla_process
#         format = "%(asctime)s - %(message)s"
#         logging.basicConfig(
#             format=format,
#             filename='carla_state.log',
#             level=logging.DEBUG
#         )

    
#     def monitor_carla_server(self):
#         while True:
#             with open("monitor_carla.txt", 'a') as f:
#                 f.write(":1111111\n")
#             if self.carla_process is None or self.carla_process.poll() is not None:
#                 with open("monitor_carla.txt", 'a') as f:
#                     f.write("222222222\n")
#                 # 重新启动服务器的逻辑
#                 # self.carla_state = self.carla_process.poll()
#                 # logging.debug('carla服务当前未运行, carla_state = ', self.carla_process.poll())

#                 self.kill_port(self.port)

#                 sleep(1)
#                 try:
#                     self.start_carla_server()
#                 except:
#                     logging.debug('monitor_carla_server重启Carla进程启动失败, carla_state = ', self.carla_process.poll())

#             sleep(1)


#     def start_carla_server(self):
#         log = open(self.log, 'a')
#         try:
#             if self.carla_process is None or self.carla_process.poll() is not None:
#                 self.carla_process = subprocess.Popen(self.carla_commend, stdout = log, stderr = log, shell=False)
#                 sleep(5)
#         except Exception as e:
#             logging.debug('Carla进程启动失败, carla_state = ', self.carla_state)


#     def kill_port(self, port):
        
#         find_port = ["lsof", "-i", ":%s" % port, "-t"]
#         result = subprocess.run(find_port, capture_output=True, text=True)
#         pids = result.stdout.strip().split("\n")
#         for pid in pids:
#             if pid:
#                 self.kill_pid(pid)


#     def kill_pid(self, pid):
#         find_kill = ["kill", "-9", pid]
#         print(find_kill)
#         result = subprocess.run(find_kill, capture_output=True, text=True)
#         print("Killed process %s" % pid)


class CarlaManage:
    """
    管理 Carla 的类，用于启动、关闭 Carla 服务器
    """
    def __init__(self):
        self.carla_state = 0
        self.port = 2000
        self.carla_process = None
        self.log = "carla_log" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ".txt"
        self.carla_commend = ["bash", "/home/dell/Soft/CARLA_0.9.13/CarlaUE4.sh"]

        format = "%(asctime)s - %(message)s"
        logging.basicConfig(
            format=format,
            filename='carla_exception.log',
            level=logging.DEBUG
        )


    def start_carla_server(self):
        try:
            log = open(self.log, 'a')
            if self.carla_process is None or self.carla_process.poll() is not None:
                self.carla_process = subprocess.Popen(self.carla_commend, stdout = log, stderr = log, shell=False)
                sleep(5)
        except Exception as e:
            logging.debug('Carla进程启动失败, carla_state = ', e)


    def stop_carla_server(self):
        self.kill_port(2000)

   
    def kill_port(self, port): 
        try:  
            find_port = ["lsof", "-i", ":%s" % port, "-t"]
            result = subprocess.run(find_port, capture_output=True, text=True)
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                if pid:
                    self.kill_pid(pid)
        except Exception as e:
            logging.debug('寻找carla pid失败', e)


    def kill_pid(self, pid):
        try:
            find_kill = ["kill", "-9", pid]
            # print(find_kill)
            # result = subprocess.run(find_kill, capture_output=True, text=True)
            subprocess.run(find_kill, capture_output=True, text=True)
        except Exception as e:
                logging.debug('kill Carla pid 失败', e)


    # def monitor_carla_server(self):
    #     # redis_conn = redis.Redis(host='localhost', port=6379, decode_responses=True)
    #     # redis_conn.set('CARLA_IS_RUNNING', 'True')
    #     while True:
    #         # 判断carla是否运行，不运行则kill掉carla进程，并重新启动
    #         if self.carla_process is None or self.carla_process.poll() is not None:

    #             self.carla_state = self.carla_process.poll()
                
    #             logging.debug('carla服务当前未运行, carla_state = ', self.carla_process.poll())

    #             self.kill_port(self.port)

    #             sleep(1)
    #             try:
    #                 self.start_carla_server()
    #             except:
    #                 logging.debug('monitor_carla_server重启Carla进程启动失败, carla_state = ', self.carla_process.poll())
        

    # def stop_carla_server(self):
    #     if self.carla_process is not None:
    #         self.carla_process.terminate()


if __name__ == "__main__":

    carlaManage = CarlaManage()
    carlaManage.start_carla_server()
    # carlaManage.monitor_carla_server()