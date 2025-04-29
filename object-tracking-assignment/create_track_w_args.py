import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def get_point_on_random_side(width, height):
    side = random.randint(0, 4)
    if side == 0:
        x = random.randint(0, width)
        y = 0
    elif side == 1:
        x = random.randint(0, width)
        y = height
    elif side == 2:
        x = 0
        y = random.randint(0, height)
    else:
        x = width
        y = random.randint(0, height)
    return x, y


def fun(x, a, b, c, d):
    return a * x + b * x ** 2 + c * x ** 3 + d


def check_track(track, width, height):
    if all(el['x'] == track[0]['x'] for el in track):
        return False
    if all(el['y'] == track[0]['y'] for el in track):
        return False
    if not all(el['x'] >= 0 and el['x'] <= width for el in track):
        return False
    if not all(el['y'] >= 0 and el['y'] <= height for el in track):
        return False
    if (2 > track[0]['x'] > (width - 2) and 2 > track[0]['y'] > (width - 2)) or (2 > track[-1]['x'] > (width - 2) and 2 > track[-1]['y'] > (width - 2)):
        return False
    return True


def add_track_to_tracks(track, tracks, id, cb_width, cb_height, random_range, bb_skip_percent):
    for i, p in enumerate(track):
        if random.random() < bb_skip_percent:
            bounding_box = []
        else:
            bounding_box = [
                              p['x'] - int(cb_width / 2) + random.randint(-random_range, random_range),
                              p['y'] - cb_height + random.randint(-random_range, random_range),
                              p['x'] + int(cb_width / 2) + random.randint(-random_range, random_range),
                              p['y'] + random.randint(-random_range, random_range)
                            ]
        if i < len(tracks):
            tracks[i]['data'].append({'cb_id': id, 'bounding_box': bounding_box,
                                      'x': p['x'], 'y': p['y'], 'track_id': None})
        else:
            tracks.append(
                {
                    'frame_id': len(tracks) + 1,
                    'data': [{'cb_id': id, 'bounding_box': bounding_box,
                              'x': p['x'], 'y': p['y'], 'track_id': None}]
                }
            )
    return tracks


def generate_tracks(tracks_amount, random_range, bb_skip_percent, width=1000, height=800, cb_width=120, cb_height=100):
    tracks = []
    i = 0

    while i < tracks_amount:
        x, y = np.array([]), np.array([])
        p = get_point_on_random_side(width, height)
        x = np.append(x, p[0])
        y = np.append(y, p[1])
        x = np.append(x, random.randint(200, width - 200))
        y = np.append(y, random.randint(200, height - 200))
        x = np.append(x, random.randint(200, width - 200))
        y = np.append(y, random.randint(200, height - 200))
        p = get_point_on_random_side(width, height)
        x = np.append(x, p[0])
        y = np.append(y, p[1])
        num = random.randint(20, 50)

        coef, _ = curve_fit(fun, x, y)
        track = [{'x': int(x), 'y': int(y)} for x, y in
                 zip(np.linspace(x[0], x[-1], num=num), fun(np.linspace(x[0], x[-1], num=num), *coef))]

        if check_track(track, width, height):
            tracks = add_track_to_tracks(track, tracks, i, cb_width, cb_height, random_range, bb_skip_percent)
            i += 1
    return tracks


def save_tracks_to_file(tracks, tracks_amount, random_range, bb_skip_percent):
    dir_name = f"obj{tracks_amount}"
    os.makedirs(dir_name, exist_ok=True)

    filename = f"{dir_name}/obj{tracks_amount}_{random_range}_{str(bb_skip_percent).replace('.','')}.py"

    with open(filename, 'w') as f:
        f.write(f"country_balls_amount = {tracks_amount}\n")
        f.write(f"track_data = {tracks}\n")

    print(f"Data saved to {filename}")


def plot_tracks(x, y, coef):
    plt.plot(x, y, 'o', label='Original points')
    plt.plot(np.linspace(x[0], x[-1]), fun(np.linspace(x[0], x[-1]), *coef), '*-')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generate track data.")
    parser.add_argument('--tracks_amount', type=int, default=5, help='Количество объектов')
    parser.add_argument('--random_range', type=int, default=2, help='На сколько пикселей рамка объектов может ложно смещаться')
    parser.add_argument('--bb_skip_percent', type=float, default=0.20, help='С какой вероятностью объект на фрейме может быть не найден детектором')
    args = parser.parse_args()

    tracks = generate_tracks(args.tracks_amount, args.random_range, args.bb_skip_percent)

    save_tracks_to_file(tracks, args.tracks_amount, args.random_range, args.bb_skip_percent)

    # print(f'country_balls_amount = {args.tracks_amount}')
    # print(f'track_data = {tracks}')

    return args.tracks_amount, tracks

if __name__ == "__main__":
    main()
