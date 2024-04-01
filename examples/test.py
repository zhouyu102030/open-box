import redis   # 导入redis 模块
from datetime import datetime, timedelta
# # 使用连接池加快链接速度
pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)

r = redis.Redis(connection_pool=pool)  


START_EXPERIMENT_TIME = datetime.fromisoformat(r.get('START_EXPERIMENT_TIME'))
print("START_EXPERIMENT_TIME = ", START_EXPERIMENT_TIME)

START_INJECT_TIME = timedelta(seconds=2) + START_EXPERIMENT_TIME
print("START_INJECT_TIME = ", START_INJECT_TIME)

r.set('START_INJECT_TIME', START_INJECT_TIME.isoformat())

print(r.get('START_INJECT'))

# # 设置键值对
# # r.set('name', 'runoob')  # 设置 name 对应的值
# # print(r['name'])
# # print(r.get('name'))  # 取出键 name 对应的值
# # print(type(r.get('name')))  # 查看类型

# # 一次性存入多个键值对
# r.mset({'k1': 'v1', 'k2': 'v2'})
# print(r.mget("k1", "k2"))   # 一次取出多个键对应的值
# # print(r.mget(['k1', 'k2']))
# # print(r.mget("k1"))

# r.set('k1', 'runoob')
# print("修改k1后的值:", r.get('k1'))


# from datetime import datetime, timedelta

# # 创建一个时间类型变量
# time_variable = datetime.strptime("2024-03-29 12:00:00", "%Y-%m-%d %H:%M:%S")

# # 加三秒
# time_variable += timedelta(seconds=3)

# # 打印结果
# print(time_variable.strftime("%Y-%m-%d %H:%M:%S"))

# from datetime import datetime, timedelta


# time = datetime.now().replace(microsecond=0)
# time1 = timedelta(seconds=2) + time

# print('time = ', time, 'time1 = ', time1)

# from datetime import datetime, timedelta

# # 从 ISO 格式的日期时间字符串创建 datetime 对象
# iso_datetime_str = datetime.now().replace(microsecond=0).isoformat()
# converted_datetime = datetime.strptime(iso_datetime_str, "%Y-%m-%dT%H:%M:%S")

# time1 = timedelta(seconds=2) + converted_datetime

# print(type(converted_datetime))  # 输出 <class 'datetime.datetime'>
# print(converted_datetime)        # 输出 2024-04-01 15:30:00
# print(time1)

