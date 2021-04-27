import datetime


def string_timestamp_hours():
    return str(datetime.datetime.now().strftime("%d_%m_%y-%H"))

def string_timestamp_minutes():
    return str(datetime.datetime.now().strftime("%d_%m_%y-%H-%M"))

def string_timestamp_seconds():
    return str(datetime.datetime.now().strftime("%d_%m_%y-%H-%M-%S"))