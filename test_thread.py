import threading
import time

def n():
    msg = input()
    print(msg)

if __name__ == '__main__':
    th = threading.Thread(target=n)
    th.daemon=True
    th.start()

    print(2)

    time.sleep(5)