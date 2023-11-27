import multiprocessing
import os
from typing import Optional


def get_memory_rss(pid: int) -> int:
    """
    Returns the physical memory used by a process.

    :param pid: process id, current one is `os.getpid()`
    :return: physical memory

    It relies on the module :epkg:`psutil`.
    """
    import psutil

    process = psutil.Process(pid)
    mem = process.memory_info().rss
    return mem


def _process_memory_spy(conn):
    # Sends the value it started.
    conn.send(-2)

    # process id to spy on
    pid = conn.recv()

    # delay between two measures
    timeout = conn.recv()

    import psutil

    process = psutil.Process(pid)

    begin = process.memory_info().rss
    max_peak = 0
    average = 0
    n_measures = 0

    conn.send(-2)

    while True:
        mem = process.memory_info().rss
        max_peak = max(mem, max_peak)
        average += mem
        n_measures += 1
        if conn.poll(timeout=timeout):
            code = conn.recv()
            if code == -3:
                break

    end = process.memory_info().rss
    average += end
    n_measures += 1
    max_peak = max(max_peak, n_measures)
    conn.send(max_peak)
    conn.send(average)
    conn.send(n_measures)
    conn.send(begin)
    conn.send(end)
    conn.close()


class MemorySpy:
    """
    Information about the spy. It class method `start`.
    Method `stop` can be called to end the measure.

    :param pid: process id  of the process to spy on
    :param delay: spy on every delay seconds
    """

    def __init__(self, pid: int, delay: float = 0.01):
        self.pid = pid
        self.delay = delay
        self.start()

    def start(self) -> "MemorySpy":
        """
        Starts another process and tells it to spy.
        """
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        self.child_process = multiprocessing.Process(
            target=_process_memory_spy, args=(self.child_conn,)
        )
        self.child_process.start()
        data = self.parent_conn.recv()
        if data != -2:
            raise RuntimeError(
                f"The child processing is supposed to send -2 not {data}."
            )
        self.parent_conn.send(self.pid)
        self.parent_conn.send(self.delay)
        data = self.parent_conn.recv()
        if data != -2:
            raise RuntimeError(
                f"The child processing is supposed to send -2 again not {data}."
            )
        return self

    def stop(self):
        """
        Stops spying on.
        """
        self.parent_conn.send(-3)
        max_peak = self.parent_conn.recv()
        average = self.parent_conn.recv()
        n_measures = self.parent_conn.recv()
        begin = self.parent_conn.recv()
        end = self.parent_conn.recv()
        self.parent_conn.close()
        self.child_process.join()
        return dict(
            max_peak=max_peak,
            average=average,
            n_measures=n_measures,
            begin=begin,
            end=end,
        )


def start_spying_on(pid: Optional[int] = None, delay: float = 0.01) -> MemorySpy:
    """
    Starts the memory spy. The function starts another
    process spying on the one sent as an argument.

    :param pid: process id to spy or the the current one.
    :param delay: delay between two measures.

    Example::

    .. code-block:: python

        from onnx_extended.memory_peak import start_spying_on

        p = start_spying_on()
        # ...
        # code to measure
        # ...
        stat = p.stop()
        print(stat)
    """
    if pid is None:
        pid = os.getpid()
    return MemorySpy(pid, delay)
