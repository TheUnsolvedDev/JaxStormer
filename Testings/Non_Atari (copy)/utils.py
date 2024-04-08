from import_packages import *


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def save_data(file=f'', data=None):
    os.makedirs(file.replace('logs.txt', ''), exist_ok=True)
    data = [round(float(dat), 4) for dat in data]
    with open(file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(data)
