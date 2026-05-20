import time


class Timer:
    def __init__(self, name=None, text="{name}: {seconds:.6f}s", logger=print):
        self.name = name or "timer"
        self.text = text
        self.logger = logger
        self.start_time = None
        self.last = 0.0

    def start(self):
        self.start_time = time.perf_counter()
        return self

    def stop(self):
        if self.start_time is None:
            self.last = 0.0
            return self.last
        self.last = time.perf_counter() - self.start_time
        self.start_time = None
        if self.logger is not None:
            self.logger(self.text.format(name=self.name, seconds=self.last))
        return self.last

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, exc_tb):
        self.stop()
        return False
