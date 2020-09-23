import os
import sys
import dill
import time
import psutil
from collections import namedtuple
from multiprocessing import Process, Manager, freeze_support, Pipe


class SignalException(Exception):
    pass


class TimeoutException(Exception):
    pass


def get_platform():
    platforms = {
        'linux1': 'Linux',
        'linux2': 'Linux',
        'darwin': 'OSX',
        'win32': 'Windows'
    }
    if sys.platform not in platforms:
        raise ValueError('Unsupported platform - %s.' % sys.platform)
    return platforms[sys.platform]


_platform = get_platform()
Returns = namedtuple('return_values', ['timeout_status', 'results'])


def wrapper_func(*args, **kwargs):
    # parse args.
    _func, _conn, _time_limit, args = args[0], args[1], args[2], args[3:]
    _func = dill.loads(_func)
    result = (False, None)

    if _platform in ['Linux', 'OSX']:
        import signal

        def handler(signum, frame):
            if signum == signal.SIGALRM:
                raise TimeoutException
            else:
                raise SignalException

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(_time_limit)
    try:
        result = (False, _func(*args, **kwargs))
    except TimeoutException:
        result = (True, None)

    finally:
        try:
            _conn.send(result)
            _conn.close()
        except:
            pass
        finally:
            p = psutil.Process()
            for child in p.children(recursive=True):
                child.kill()


def time_limit(func, time, *args, **kwargs):
    freeze_support()
    parent_conn, child_conn = Pipe(False)

    func = dill.dumps(func)
    args = [func] + [child_conn] + [time] + list(args)

    p = Process(target=wrapper_func, args=tuple(args), kwargs=kwargs)
    p.start()

    p.join(time)
    if p.is_alive():
        p.terminate()
        return Returns(timeout_status=True, results=None)
    result = parent_conn.recv()
    parent_conn.close()
    if result[0] is True:
        return Returns(timeout_status=True, results=None)
    return Returns(timeout_status=False, results=result[1])


@DeprecationWarning
def ps_time_limit(func, args, kwargs, time):
    Returns = namedtuple('return_values', ['status', 'results'])
    with Manager() as manager:
        result_container = manager.list()
        args = list(args) + [result_container]
        p1 = Process(target=func, args=tuple(args), kwargs=kwargs)
        p1.start()
        # def on_terminate(proc):
        #     print("process {} terminated with exit code {}".format(proc, proc.returncode))
        procs = psutil.Process().children()
        # print(p1.pid)
        # print(os.getpid(), psutil.Process().pid)
        # print(procs)
        procs = [p for p in procs if p.pid == p1.pid]
        # print(procs)
        gone, alive = psutil.wait_procs(procs, timeout=time)
        result = list(result_container)

    if len(alive) > 0:
        for p in alive:
            p.kill()
        return Returns(status=False, results=None)
    return Returns(status=True, results=result)


def test_func(*args, **kwargs):
    import time
    n = args[0]
    time.sleep(n)
    return n * n


if __name__ == "__main__":
    a = (3)
    b = dict()
    res = time_limit(test_func, 5, a, b)
    # res = ps_time_limit(test, a, b, 1)
    print(res)