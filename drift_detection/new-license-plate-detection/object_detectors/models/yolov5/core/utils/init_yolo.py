
"""
utils/initialization
"""

import contextlib
import platform
import threading


def emojis(str=''):

    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str


class TryExcept(contextlib.ContextDecorator):

    def __init__(self, msg=''):
        self.msg = msg

    def __enter__(self):
        pass

    def __exit__(self, exc_type, value, traceback):
        if value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True


def threaded(func):

    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread

    return wrapper


def join_threads(verbose=False):

    main_thread = threading.current_thread()
    for t in threading.enumerate():
        if t is not main_thread:
            if verbose:
                print(f'Joining thread {t.name}')
            t.join()


def notebook_init(verbose=True):

    print('Checking setup...')

    import os
    import shutil

    from ultralytics.yolo.utils.checks import check_requirements

    from models.yolov5.core.utils.general import check_font, is_colab
    from models.yolov5.core.utils.torch_utils import select_device

    check_font()

    import psutil

    if check_requirements('wandb', install=False):
        os.system('pip uninstall -y wandb')
    if is_colab():
        shutil.rmtree('/content/sample_data', ignore_errors=True)


    display = None
    if verbose:
        gb = 1 << 30
        ram = psutil.virtual_memory().total
        total, used, free = shutil.disk_usage('/')
        with contextlib.suppress(Exception):
            from IPython import display
            display.clear_output()
        s = f'({os.cpu_count()} CPUs, {ram / gb:.1f} GB RAM, {(total - free) / gb:.1f}/{total / gb:.1f} GB disk)'
    else:
        s = ''

    select_device(newline=False)
    print(emojis(f'Setup complete ??{s}'))
    return display
