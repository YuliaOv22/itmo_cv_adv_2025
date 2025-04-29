import asyncio
import base64
import glob
import importlib
import io
import os
import re
import subprocess
import sys
import traceback
from io import BytesIO

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from fastapi import FastAPI, WebSocket
from metrics import TrackingMetrics
from PIL import Image
from starlette.websockets import WebSocketDisconnect, WebSocketState

MAX_DISTANCE_THRESHOLD = 400
INITIAL_TTL = 6

app = FastAPI(title='Tracker assignment')
imgs = glob.glob('imgs/*')



def euclidean_distance(x : np.ndarray, y : np.ndarray):
    return np.linalg.norm(x - y)


def calculate_centroid(bbox: list):
    x_1, y_1, x_2, y_2 = bbox
    x_c = x_1 + int((x_2 - x_1) / 2)
    y_c = y_1 + int((y_2 - y_1) / 2)
    return  np.array([x_c, y_c])


def find_best_match(centroid, available_tracks):
    best_id = None
    best_distance = MAX_DISTANCE_THRESHOLD
    
    for track_id, track_info in available_tracks.items():
        if track_info['centroid'] is not None:
            dist = euclidean_distance(np.array(track_info['centroid']), np.array(centroid))
            if dist < best_distance:
                best_distance = dist
                best_id = track_id
    return best_id, best_distance


def tracker_soft(el):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов

    Ограничения:
    - необходимо использовать как можно меньше ресурсов (представьте, что
    вы используете embedded устройство, например Raspberri Pi 2/3).
    - значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме
    """

    if el['frame_id'] == 1:
        tracker_soft.prev_tracks = {}
        tracker_soft.next_id = 0
        tracker_soft.used_ids_history = set()

        for x in el['data']:
            if x['bounding_box']:
                cent = calculate_centroid(x['bounding_box'])
                tracker_soft.prev_tracks[tracker_soft.next_id] = {
                    'centroid': cent,
                    'ttl': INITIAL_TTL
                }
            else:
                tracker_soft.prev_tracks[tracker_soft.next_id] = {
                    'centroid': None,
                    'ttl': INITIAL_TTL
                }

            x['track_id'] = tracker_soft.next_id
            tracker_soft.used_ids_history.add(tracker_soft.next_id)
            tracker_soft.next_id += 1
        return el

    current_tracks = {}
    used_prev_ids = set()

    for x in el['data']:
        if x['bounding_box']:
            cent = calculate_centroid(x['bounding_box'])
            best_id = None

            avail_prev = {k: v for k, v in tracker_soft.prev_tracks.items() 
                         if k not in used_prev_ids and v['centroid'] is not None}
            best_id, _ = find_best_match(cent, avail_prev)

            if best_id is not None:
                x['track_id'] = best_id
                used_prev_ids.add(best_id)
                current_tracks[best_id] = {
                    'centroid': cent,
                    'ttl': INITIAL_TTL
                }
            else:
                tracker_soft.next_id = max(list(tracker_soft.used_ids_history)) + 1
                
                x['track_id'] = tracker_soft.next_id
                used_prev_ids.add(tracker_soft.next_id)
                tracker_soft.used_ids_history.add(tracker_soft.next_id)
                current_tracks[tracker_soft.next_id] = {
                    'centroid': cent,
                    'ttl': INITIAL_TTL
                }
                tracker_soft.next_id += 1

        else:
            available_tracks = {k: v for k, v in tracker_soft.prev_tracks.items() 
                              if k not in used_prev_ids and v['ttl'] > 0}
            
            if available_tracks:
                best_id = max(available_tracks.items(), key=lambda item: item[1]['ttl'])[0]
                x['track_id'] = best_id
                used_prev_ids.add(best_id)
                tracker_soft.used_ids_history.add(best_id)
                current_tracks[best_id] = {
                    'centroid': None,
                    'ttl': available_tracks[best_id]['ttl'] - 1
                }

    for track_id, track_info in tracker_soft.prev_tracks.items():
        if track_id not in current_tracks:
            if track_info['ttl'] > 1:
                current_tracks[track_id] = {
                    'centroid': track_info['centroid'],
                    'ttl': track_info['ttl'] - 1
                }

    return el


def xyxy2xywh(bbox, frame_shape=None):
    x1, y1, x2, y2 = bbox

    if frame_shape:
        h_img, w_img, _ = frame_shape
        x1 = max(0, min(x1, w_img - 1))
        x2 = max(0, min(x2, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        y2 = max(0, min(y2, h_img - 1))

    w = max(0, x2 - x1)
    h = max(0, y2 - y1)

    return [x1, y1, w, h]


def tracker_strong(tracker, el, dir_frames):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов, скриншоты прогона

    Ограничения:
    - вы можете использовать любые доступные подходы, за исключением
    откровенно читерных, как например захардкодить заранее правильные значения
    'track_id' и т.п.
    - значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме

    P.S.: если вам нужны сами фреймы, измените в index.html значение make_screenshot
    на true и воспользуйтесь нижним закомментированным кодом в этом файле для первого прогона, 
    на повторном прогоне можете читать сохраненные фреймы из папки
    и по координатам вырезать необходимые регионы.
    """
        
    if el['frame_id'] == 1:
        tracker.deepsort = DeepSort(
            max_age=12,
            n_init=3,
            nms_max_overlap=0.8,
            max_cosine_distance=0.5,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=False,
            embedder_gpu=True,
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None
        )
        tracker.extra_tracks = {}
        tracker.next_id = 0
        tracker.used_ids_history = set()

    frame_img = cv2.imread(f'{dir_frames}/frame_{el["frame_id"]}.png')
    frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
   
    # Подготовка детекций в нужном формате
    detections = []
    index_to_obj = []
    for idx, x in enumerate(el['data']):
        if x['bounding_box']:
            detections.append((xyxy2xywh(x['bounding_box'], frame_shape=frame_img.shape), 1, 1))  # bbox, confidence, class_id
            index_to_obj.append(idx)

    #  Ищем треки
    if detections:
        found_tracks = tracker.deepsort.update_tracks(detections, frame=frame_img)
    else:
        found_tracks = []

    # Сопоставляем треки объектам с bbox
    detections_track = {item.track_id: calculate_centroid(item.to_ltrb()) for item in found_tracks if item.is_confirmed()}
    current_ids = set()

    for idx in index_to_obj:
        x = el['data'][idx]
        if len(detections_track) > 0:
            centr_el = calculate_centroid(x['bounding_box'])
            best_id, _ = min(detections_track.items(),
                             key=lambda item: euclidean_distance(item[1], centr_el))
            x['track_id'] = best_id
            current_ids.add(best_id)
            tracker.used_ids_history.add(best_id)
            del detections_track[best_id]

    # Объекты без bbox
    used_extra_ids = set()
    for x in el['data']:
        if x['bounding_box']:
            continue  # Уже обработано выше

        # Пробуем сопоставить с активными extra треками
        available_tracks = {k: v for k, v in tracker.extra_tracks.items() if v['ttl'] > 0 and k not in used_extra_ids}
        if available_tracks:
            best_id = max(available_tracks.items(), key=lambda item: item[1]['ttl'])[0]
            x['track_id'] = best_id
            tracker.extra_tracks[best_id]['ttl'] -= 1
            used_extra_ids.add(best_id)
            current_ids.add(best_id)
        else:
            # Новый track_id
            # if tracker.used_ids_history:
            #     new_id = max([int(x) for x in tracker.used_ids_history]) + 1
            # else:
            #     new_id = 0
            new_id = tracker.next_id
            tracker.next_id += 1
            x['track_id'] = new_id
            tracker.extra_tracks[new_id] = {'ttl': INITIAL_TTL - 1}
            tracker.used_ids_history.add(new_id)
            used_extra_ids.add(new_id)
            current_ids.add(new_id)

    for track_id in list(tracker.extra_tracks.keys()):
        if track_id not in current_ids:
            tracker.extra_tracks[track_id]['ttl'] -= 1
            if tracker.extra_tracks[track_id]['ttl'] <= 0:
                del tracker.extra_tracks[track_id]

    return el


def save_results_to_md(results: list, file_path: str) -> None:
    file_exists = os.path.exists(file_path)

    with open(file_path, 'a' if file_exists else 'w') as file:
        if not file_exists:
            file.write("| Объекты | random_range | bb_skip_percent |  Average Track Coverage | ID Switch Count | Mismatch Ratio |\n")
            file.write("|:-------:|:------------:|:---------------:|:------------:|:---------------:|:--------------:|\n")

        for result in results:
            file.write(f"| {result['objects']} | {result['random_range']} | {result['bb_skip_percent']} | {result['Average Track Coverage']} | {result['ID Switch Count']} | {result['Mismatch Ratio']} |\n")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print('Accepting client connection...')
    await websocket.accept()

    tracks_amount = 5
    # tracks_amount = 10
    # tracks_amount = 20


    random_range = 2
    # random_range = 5

    bb_skip_percent = 0.2
    # bb_skip_percent = 0.4


    module_name = f'obj{tracks_amount}.obj{tracks_amount}_{random_range}_{str(bb_skip_percent).replace(".","")}'
    dir = f'./obj{tracks_amount}/obj{tracks_amount}_{random_range}_{str(bb_skip_percent).replace(".","")}'
    
    print(f"Importing module: {module_name}")
    mod = importlib.import_module(module_name)

    country_balls_amount = mod.country_balls_amount
    track_data = mod.track_data
    print(f"Running tracking for country_balls_amount = {country_balls_amount}")

    country_balls = [{'cb_id': x, 'img': imgs[x % len(imgs)]} for x in range(country_balls_amount)]
    print('Started')

    tracker = DeepSort()
    metrics = TrackingMetrics()

    await websocket.send_text(str(country_balls))

    for el in track_data:
        await asyncio.sleep(1)
        # el = tracker_soft(el)

        el = tracker_strong(tracker, el, dir)
        # print(len(el['data']), el['data'])
        await websocket.send_json(el)

        metrics.add_frame(el)

    metrics_dict = metrics.calculate_metrics()
    result = {
        'objects': tracks_amount,
        'random_range': random_range,
        'bb_skip_percent': bb_skip_percent,
        'Average Track Coverage': metrics_dict['avg_track_coverage'],
        'ID Switch Count': metrics_dict['id_switch_count'],
        'Mismatch Ratio': metrics_dict['mismatch_ratio']
    }

    results = [result]

    # save_results_to_md(results, 'strong_tracking_results.md')
    # save_results_to_md(results, 'soft_tracking_results_new.md')


    print(f'Metrics for {tracks_amount} tracks, random_range = {random_range}, bb_skip_percent = {bb_skip_percent}')
    metrics.report()

    await asyncio.sleep(0.5)
    print('Bye..')
    await websocket.close()

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     print('Accepting client connection...')
#     await websocket.accept()
#     await websocket.receive_text()
#     # отправка служебной информации для инициализации объектов
#     # класса CountryBall на фронте

#     tracks_amount_values = [5, 10, 20]
#     random_range_values = [2, 5]
#     bb_skip_percent_values = [0.0, 0.2, 0.4]
#     for tracks_amount in tracks_amount_values:
#         for random_range in random_range_values:
#             for bb_skip_percent in bb_skip_percent_values:

#                 module_name = f'obj{tracks_amount}.obj{tracks_amount}_{random_range}_{str(bb_skip_percent).replace(".","")}'
#                 mod = importlib.import_module(module_name)
#                 country_balls_amount = mod.country_balls_amount
#                 track_data = mod.track_data

#                 country_balls = [{'cb_id': x, 'img': imgs[x % len(imgs)]} for x in range(country_balls_amount)]
#                 print('Started')

#                 dir = f"./{module_name.replace('.','/')}/"
#                 if not os.path.exists(dir):
#                     os.makedirs(dir)

#                 await websocket.send_text(str(country_balls))
#                 for el in track_data:
#                     await asyncio.sleep(0.5)
#                     image_data = await websocket.receive_text()
#                     # print(image_data)
#                     try:
#                         image_data = re.sub('^data:image/.+;base64,', '', image_data)
#                         image = Image.open(BytesIO(base64.b64decode(image_data)))
#                         image = image.resize((1000, 800), Image.Resampling.LANCZOS)
#                         frame_id = el['frame_id'] - 1
#                         image.save(f"{dir}/frame_{frame_id}.png")
#                         # print(image)
#                     except Exception as e:
#                         print(e)
                
#                     # отправка информации по фрейму
#                     await websocket.send_json(el)

#                 await websocket.send_json(el)
#                 await asyncio.sleep(0.5)
#                 image_data = await websocket.receive_text()
#                 try:
#                     image_data = re.sub('^data:image/.+;base64,', '', image_data)
#                     image = Image.open(BytesIO(base64.b64decode(image_data)))
#                     image = image.resize((1000, 800), Image.Resampling.LANCZOS)
#                     image.save(f"{dir}/frame_{el['frame_id']}.png")
#                 except Exception as e:
#                     print(e)

#                 print('Bye..')
