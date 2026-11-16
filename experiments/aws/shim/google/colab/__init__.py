"""Stand-in for google.colab so the Colab notebook runs unmodified on EC2 (PYTHONPATH=shim).

drive.mount -> local dir, userdata.get -> environment variable (from secrets.env).
"""
import os


class drive:
    @staticmethod
    def mount(path, **_):
        os.makedirs(os.path.join(path, "MyDrive"), exist_ok=True)


class userdata:
    @staticmethod
    def get(key):
        return os.environ.get(key)
