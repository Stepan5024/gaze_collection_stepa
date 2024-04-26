import pathlib
from argparse import ArgumentParser
from collections import defaultdict
import os
import sys
import h5py
import cv2
import pandas as pd
import ast
import numpy as np
from utils import show_point_on_screen, get_monitor_dimensions
from webcam import WebcamSource

WINDOW_NAME = 'data collection'


def main(base_path: str, monitor_mm="(400, 250)", monitor_pixels="(1920, 1080)", pxx:str="p00", day:str="day01"):
    pathlib.Path(f'{base_path}\/').mkdir(parents=True, exist_ok=True)

    source = WebcamSource()
    # Получить следующий кадр от веб-камеры
    image = next(source)

  
    if monitor_mm is None or monitor_pixels is None:
        monitor_mm, monitor_pixels = get_monitor_dimensions()
        if monitor_mm is None or monitor_pixels is None:
            raise ValueError('Please supply monitor dimensions manually as they could not be retrieved.')
    print(f'Found default monitor of size {monitor_mm[0]}x{monitor_mm[1]}mm and {monitor_pixels[0]}x{monitor_pixels[1]}px.')

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    collected_data = defaultdict(list)
    # Добавляем подпапку к base_path
    full_path = os.path.join(base_path, day)
    if not os.path.exists(full_path):
        os.makedirs(full_path, exist_ok=True)
        print("Папки были успешно созданы по пути:", full_path)

    while True:
        ( center, time_till_capture, 
         quit_requested, gaze_pitch, gaze_yaw, paths_list) = show_point_on_screen(WINDOW_NAME, full_path, monitor_pixels, source)
        print(f"paths_list {paths_list}")
        if  time_till_capture is not None:
            for file_name in paths_list:
                if file_name is not None:
                    print(f'type_file_name {type(file_name[0])}')
                    print(f'type_center {type(center)}')
                    print(f'type_gaze_pitch {type(gaze_pitch)}')
                    print(f'type_gaze_yaw {type(gaze_yaw)}')
                    print(f'type_monitor_pixels {type(monitor_pixels)}')
                    collected_data['file_name_base'].append(str(file_name[0]))
                    #point_on_screen
                    collected_data['gaze_location'].append(center)
                    collected_data['gaze_pitch'].append(gaze_pitch)
                    collected_data['gaze_yaw'].append(gaze_yaw)
                    #collected_data['time_till_capture'].append(time_till_capture)
                    #collected_data['monitor_mm'].append(monitor_mm)
                    #monitor_pixels
                    collected_data['screen_size'].append(monitor_pixels)
        if quit_requested:
            break
        if cv2.waitKey(500) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    # После окончания сбора данных записываем их в файл
    df_new = pd.DataFrame(collected_data)
    file_path = os.path.join(os.path.dirname(base_path), 'data.csv')


    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)
        if df_existing.empty:
            last_index = 0  # or 1, depending on how you want to start indexing
        else:
        # Ensure last_index is an integer. NaN or float values are coerced to 0.
            last_index = int(df_existing.index.max() + 1)
        #last_index = df_existing.index.max() + 1
        df_new.index = range(last_index, last_index + len(df_new))
        # Файл существует, дозаписываем данные без заголовков
        df_new.to_csv(file_path, mode='a', header=False, index=True)
    else:
        # Файл не существует, создаем новый файл с заголовками
        df_new.to_csv(file_path, mode='w', header=True, index=True)
    print('Запись файла!')
    # Сохранение данных DataFrame в файл HDF5
    h5_file_path = os.path.join(os.path.dirname(base_path), 'data.h5')
    def safe_convert_to_tuple(x):
        if isinstance(x, str):
            return ast.literal_eval(x)
        return x
    print(df_new.columns)
    if 'gaze_location' in df_new.columns:
        df_new['gaze_location'] = df_new['gaze_location'].apply(safe_convert_to_tuple)
        df_new['screen_size'] = df_new['screen_size'].apply(safe_convert_to_tuple)
    else:
        print("Column 'gaze_location' does not exist.")
    # Открываем файл HDF5 для записи
    with h5py.File(h5_file_path, 'a') as h5file:
        # Обрабатываем столбцы, которые являются строками
        # Создаем специальный тип данных для строк переменной длины
         
        dt_str = h5py.special_dtype(vlen=str)  # Создаем специальный тип данных для строк
        # Если набор данных уже существует, читаем его и добавляем новые данные
        if 'file_name_base' in h5file:
            data = list(h5file['file_name_base'])
            data.extend(df_new['file_name_base'].astype('S'))
            del h5file['file_name_base']
            h5file.create_dataset('file_name_base', data=data, dtype=dt_str)
        else:
            h5file.create_dataset('file_name_base', data=df_new['file_name_base'].astype('S'), dtype=dt_str)

        # Аналогично обрабатываем остальные столбцы
        for column in ['gaze_pitch', 'gaze_yaw']:
            if column in h5file:
                data = h5file[column][:]
                new_data = df_new[column].to_numpy(dtype=np.float32)
                data = np.concatenate((data, new_data))
                del h5file[column]
                h5file.create_dataset(column, data=data)
            else:
                h5file.create_dataset(column, data=df_new[column].to_numpy(dtype=np.float32))

        # Обрабатываем столбцы с числовыми значениями
        # Для столбцов, содержащих кортежи
        for column in ['gaze_location', 'screen_size']:
            if column in h5file:
                data = h5file[column][:]
                new_data = np.array(df_new[column].tolist(), dtype=np.int32)
                data = np.concatenate((data, new_data))
                del h5file[column]
                h5file.create_dataset(column, data=data)
            else:
                data = np.array(df_new[column].tolist(), dtype=np.int32)
                h5file.create_dataset(column, data=data)

    print(f"Data saved to {h5_file_path}")
    cv2.destroyAllWindows()
    sys.exit()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, default='./data/p02')
    parser.add_argument("--monitor_mm", type=str, default="400,250")
    parser.add_argument("--monitor_pixels", type=str, default="1920,1080")
    parser.add_argument("--pxx", type=str, default="p02")
    parser.add_argument("--day", type=str, default="day01")
    args = parser.parse_args()

    if args.monitor_mm is not None:
        args.monitor_mm = tuple(map(int, args.monitor_mm.split(',')))
    if args.monitor_pixels is not None:
        args.monitor_pixels = tuple(map(int, args.monitor_pixels.split(',')))

    main(args.base_path, args.monitor_mm, args.monitor_pixels, args.pxx, args.day)
