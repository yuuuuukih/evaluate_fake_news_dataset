import functools
import time

# decorator
def measure_exe_time(func):
    functools.wraps(func)
    def _wrapper(*args, **keywords):
        start_time = time.time()

        v = func(*args, **keywords)

        end_time = time.time()
        exe_time = end_time - start_time
        hours, remainder = divmod(exe_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_time = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        print(f"The execution time of the {func.__name__} function is {formatted_time}.")

        return v
    return _wrapper