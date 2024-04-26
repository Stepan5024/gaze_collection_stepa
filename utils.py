import random
import sys
import time
from datetime import datetime
from enum import Enum

from batch_face.face_detection import RetinaFace
from batch_face import drawLandmark_multiple, LandmarkPredictor

import pyautogui
import dlib
import os
import cv2
import numpy as np
from typing import Tuple, Union
from threading import Thread
from webcam import WebcamSource
from l2cs import Pipeline
import torch
from pathlib import Path
import time  



def get_monitor_dimensions() -> Union[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[None, None]]:
    """
    Get monitor dimensions from Gdk.
    from on https://github.com/NVlabs/few_shot_gaze/blob/master/demo/monitor.py
    :return: tuple of monitor width and height in mm and pixels or None
    """
    try:
        import pgi

        pgi.install_as_gi()
        import gi.repository

        gi.require_version('Gdk', '3.0')
        from gi.repository import Gdk

        display = Gdk.Display.get_default()
        screen = display.get_default_screen()
        default_screen = screen.get_default()
        num = default_screen.get_number()

        h_mm = default_screen.get_monitor_height_mm(num)
        w_mm = default_screen.get_monitor_width_mm(num)

        h_pixels = default_screen.get_height()
        w_pixels = default_screen.get_width()

        return (w_mm, h_mm), (w_pixels, h_pixels)

    except ModuleNotFoundError:
        return None, None


FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.5
TEXT_THICKNESS = 2


class TargetOrientation(Enum):
    UP = 82
    DOWN = 84
    LEFT = 81
    RIGHT = 83


def create_image(monitor_pixels: Tuple[int, int], center=(0, 0), circle_scale=1., orientation=TargetOrientation.RIGHT, target='E') -> Tuple[np.ndarray, float, bool]:
    """
    Create image to display on screen.

    :param monitor_pixels: monitor dimensions in pixels
    :param center: center of the circle and the text
    :param circle_scale: scale of the circle
    :param orientation: orientation of the target
    :param target: char to write on image
    :return: created image, new smaller circle_scale and bool that indicated if it is th last frame in the animation
    """
    width, height = monitor_pixels

    if orientation == TargetOrientation.LEFT or orientation == TargetOrientation.RIGHT:
        img = np.zeros((height, width, 3), np.float32)

        if orientation == TargetOrientation.LEFT:
            center = (width - center[0], center[1])

        end_animation_loop = write_text_on_image(center, circle_scale, img, target)

        if orientation == TargetOrientation.LEFT:
            img = cv2.flip(img, 1)
    else:  # TargetOrientation.UP or TargetOrientation.DOWN
        img = np.zeros((width, height, 3), np.float32)
        center = (center[1], center[0])

        if orientation == TargetOrientation.UP:
            center = (height - center[0], center[1])

        end_animation_loop = write_text_on_image(center, circle_scale, img, target)

        if orientation == TargetOrientation.UP:
            img = cv2.flip(img, 1)

        img = img.transpose((1, 0, 2))

    return img / 255, circle_scale * 0.9, end_animation_loop


def write_text_on_image(center: Tuple[int, int], circle_scale: float, img: np.ndarray, target: str):
    """
    Write target on image and check if last frame of the animation.

    :param center: center of the circle and the text
    :param circle_scale: scale of the circle
    :param img: image to write data on
    :param target: char to write
    :return: True if last frame of the animation
    """
    text_size, _ = cv2.getTextSize(target, FONT, TEXT_SCALE, TEXT_THICKNESS)
    cv2.circle(img, center, int(text_size[0] * 5 * circle_scale), (32, 32, 32), -1)
    text_origin = (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2)

    end_animation_loop = circle_scale < random.uniform(0.1, 0.5)
    if not end_animation_loop:
        cv2.putText(img, target, text_origin, FONT, TEXT_SCALE, (17, 112, 170), TEXT_THICKNESS, cv2.LINE_AA)
    else:
        cv2.putText(img, target, text_origin, FONT, TEXT_SCALE, (252, 125, 11), TEXT_THICKNESS, cv2.LINE_AA)

    return end_animation_loop


def is_cursor_in_circle(cursor_pos: Tuple[int, int], center: Tuple[int, int], monitor_pixels: Tuple[int, int]) -> bool:
    """
    Check if the cursor is within 20% of the center of the circle.

    :param cursor_pos: Current cursor position (x, y)
    :param center: Center of the circle (x, y)
    :param monitor_pixels: monitor dimensions in pixels (width, height)
    :return: True if cursor is within the area, False otherwise
    """
    # Calculate 20% of the shortest monitor dimension as the "radius"
    percent = 0.2
    radius = percent * min(monitor_pixels) / 2

    # Calculate the distance between the cursor and the center of the circle
    distance = ((cursor_pos[0] - center[0]) ** 2 + (cursor_pos[1] - center[1]) ** 2) ** 0.5

    # Check if the cursor is within 20% of the radius from the center
    return distance <= radius

def get_random_position_on_screen(monitor_pixels: Tuple[int, int]) -> Tuple[int, int]:
    """
    Get random valid position on monitor.

    :param monitor_pixels: monitor dimensions in pixels
    :return: tuple of random valid x and y coordinated on monitor
    """
    return int(random.uniform(0, 1) * monitor_pixels[0]), int(random.uniform(0, 1) * monitor_pixels[1])


def save_eyes(frame, landmarks, base_path, file_name, paths_list):
    # Индексы точек для левого и правого глаз в массиве из 68 ключевых точек
    left_eye_indices = slice(36, 42)
    right_eye_indices = slice(42, 48)

    # Вызываем функцию extract_eye_area для каждого глаза
    left_eye_area = extract_eye_area(frame, landmarks[left_eye_indices])
    right_eye_area = extract_eye_area(frame, landmarks[right_eye_indices])

    # Приводим размеры областей глаз к одинаковым
    left_eye_area_resized, right_eye_area_resized = resize_eye(left_eye_area, right_eye_area)

    # Сохраняем изображения областей левого и правого глаз
    paths_list.append(save_image_and_get_paths(base_path, file_name, 
                                               "left_eye", left_eye_area_resized))
    paths_list.append(save_image_and_get_paths(base_path, file_name, 
                                               "right_eye", right_eye_area_resized))
def resize_eye(eye1, eye2):
    # Находим максимальные ширину и высоту среди двух изображений
    max_height = 64
    max_width = 96
    
    # Изменяем размер обеих областей глаз до максимального размера
    eye1_resized = cv2.resize(eye1, (max_width, max_height), interpolation=cv2.INTER_AREA)
    eye2_resized = cv2.resize(eye2, (max_width, max_height), interpolation=cv2.INTER_AREA)
    
    return eye1_resized, eye2_resized

def resize_face(face):
    # Находим максимальные ширину и высоту среди двух изображений
    max_height = 96
    max_width = 96
    
    # Изменяем размер обеих областей глаз до максимального размера
    face = cv2.resize(face, (max_width, max_height), interpolation=cv2.INTER_AREA)
    
    return face

def extract_eye_area(image, eye_points):
    # Увеличиваем область вокруг глаз
    expansion_factor = 0.6
    eye_points = np.array(eye_points).astype(np.int32)
    print(f"eye_points {eye_points}")
    x_coords, y_coords = eye_points[:, 0], eye_points[:, 1]
    x_min, y_min = np.min(x_coords), np.min(y_coords)
    x_max, y_max = np.max(x_coords), np.max(y_coords)
    width, height = x_max - x_min, y_max - y_min
    x_min -= int(expansion_factor * width / 2)
    y_min -= int(expansion_factor * height / 2)
    x_max += int(expansion_factor * width / 2)
    y_max += int(expansion_factor * height / 2)

    # Убедимся, что координаты не выходят за пределы изображения
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(image.shape[1] - 1, x_max), min(image.shape[0] - 1, y_max)

    # Вырезаем и возвращаем увеличенную область глаз
    eye_area = image[y_min:y_max, x_min:x_max]
    return eye_area


def save_image_and_get_paths(base_path, file_name, postfix, image):
     # Список значений postfix, при которых функция не будет выполнять сохранение
    skip_postfixes = ['left_eye', 'right_eye', 'landmark']
    # Создаем поддиректорию, если она еще не существует
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    # Разделяем путь на части
    path_parts = os.path.normpath(base_path).split(os.sep)
    # Извлекаем нужные части пути (например, 'p00/day01')
    # Предполагается, что 'p00' и 'day01' всегда находятся в конце пути
    specific_path = '/'.join(path_parts[-2:])
    # Строим полный путь файла с расширением
    full_file_path_new = os.path.join(specific_path, f"{file_name}-{postfix}.jpg")
    full_file_path = os.path.join(base_path, f"{file_name}-{postfix}.jpg")

    # Сохраняем изображение
    cv2.imwrite(full_file_path, image)

    # Удаляем расширение файла и добавляем измененный путь в список
    modified_path = os.path.splitext(full_file_path_new)[0]
    modified_path = modified_path.replace("\\", "/")
    print(f"modified_path {modified_path}")
    # Возвращаем список, содержащий один измененный путь
    # Проверяем, содержится ли postfix в списке skip_postfixes
    if postfix in skip_postfixes:
        print(f"Skipping saving for postfix '{postfix}'.")
        return
    return [modified_path]

def save_faces_from_frame(frame, base_path, file_name, paths_list):
    """
    Обнаруживает лица в кадре и сохраняет их в указанной директории.
    
    :param frame: Изображение для обнаружения лиц.
    :param base_path: Базовый путь для сохранения обнаруженных лиц.
    :param file_name: Имя файла для сохраненных изображений лиц.
    """
    # Преобразуем изображение из BGR (OpenCV) в RGB (face_recognition)
    rgb_frame = frame[:, :, ::-1]
    detector = RetinaFace(-1)
    predictor = LandmarkPredictor(-1)
    boxes = []
    faces = detector(rgb_frame, cv=True) # set cv to False for rgb input, the default value of cv is False
        # Проверяем, обнаружены ли лица
    if len(faces) == 0:
        print("NO face is detected!")
        return
    results = predictor(faces, frame, from_fd=True)
    # Рисуем ключевые точки на каждом лице
    for face, landmarks in zip(faces, results):
        print(f"face {face}")
        print(f"face[0] {face[0]}")
        print(f"landmarks {landmarks}")
        print(f"landmarks {len(landmarks)}")
        img = drawLandmark_multiple(frame, face[0], landmarks)
        print(f"results[0][1] {results[0][1]}")
        save_eyes(frame, landmarks, base_path, file_name, paths_list)
        img = resize_face(img)

    paths_list.append(save_image_and_get_paths(base_path, file_name, "landmark", img))

    for face in faces:
    # Извлекаем box, landmarks, и score из текущего лица
        box, landmarks, score = face
        print(f"box {box}")
        # Добавляем координаты прямоугольника (box) текущего лица в список boxes
        boxes.append(box)
    box, landmarks, score = faces[0]
    print(f"boxes {boxes}")
    print(f"len faces {len(faces)}")
    print(f"path {base_path}")
    print(f"box {box}")
    # Обнаруживаем лица
    
    for face in enumerate(faces):
        # Преобразуем координаты к целочисленному типу
        print(f"face {face}")  # Отладочная печать для понимания структуры данных
        _, (box, landmarks, score) = face
        # Перед использованием координат для вырезания, убедимся, что они целочисленные
        x1, y1, x2, y2 = box.astype(int)
        # Вырезаем изображение лица из фрейма
        face_image = frame[y1:y2, x1:x2]
        face_image = resize_face(face_image)
        # Здесь вы можете использовать face_image для дальнейших действий, например, сохранить в файл
        paths_list.append(save_image_and_get_paths(base_path, file_name, "full_face", face_image))

        # Предположим, что image - это ваше изображение, а landmarks - список всех ключевых точек
        print(f"landmarks {landmarks}")
        print(f"landmarks {len(landmarks)}")
    return paths_list
        

# Укажите путь к директории с моделями
CWD = Path('C:\\Users\\bokar\\Documents\\gaze_collection_stepa')

# Инициализация пайплайна с указанием пути к весам модели и архитектуре
gaze_pipeline = Pipeline(
    weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
    arch='ResNet50',
    device=torch.device('cpu')  # или torch.device('cuda') для использования GPU
)


def show_point_on_screen(window_name: str, base_path: str, 
                         monitor_pixels: Tuple[int, int], 
                         source: WebcamSource) -> Tuple[str, Tuple[int, int], float]:
    """
    Show one target on screen, full animation cycle. Return collected data if data is valid

    :param window_name: name of the window to draw on
    :param base_path: path where to save the image to
    :param monitor_pixels: monitor dimensions in pixels
    :param source: webcam source
    :return: collected data otherwise None
    """
    paths_list = []
    def process_image(image, results_container):
        """
        Function to be executed in a separate thread for image processing.
        """
        results = gaze_pipeline.step(image)
        results_container.append(results)
    # Initialize variables
    results_container = []
    circle_scale = 1.
    center = get_random_position_on_screen(monitor_pixels)
    end_animation_loop = False
    orientation = random.choice(list(TargetOrientation))

    file_name = None
    time_till_capture = None
    quit_requested = False
    gaze_pitch = None
    gaze_yaw = None
    while not end_animation_loop:
        image, circle_scale, end_animation_loop = create_image(monitor_pixels, center, circle_scale, orientation)
        cv2.imshow(window_name, image)

        for _ in range(5):  # workaround to not speed up the animation when buttons are pressed
            if cv2.waitKey(50) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                quit_requested = True
                break  # Выходим из цикла for
        if quit_requested:
            break  # Выходим из цикла while
    if end_animation_loop:
        
        file_name = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        start_time_color_change = time.time()
 
        while time.time() - start_time_color_change < 0.5:
           # Get the current position of the cursor
            cursor_pos = pyautogui.position()
            print(f'cursor_pos = {cursor_pos} center {center}')
            # Check if the cursor is within the defined area of the circle
            if is_cursor_in_circle(cursor_pos, center, monitor_pixels):
                print("The cursor is within 20% of the center of the circle.")
                image = next(source)
                paths_list = save_faces_from_frame(image, base_path, file_name, paths_list)
                 
                # Загрузка изображения
                
                # Start the thread for image processing
                thread = Thread(target=process_image, args=(image, results_container))
                thread.start()
                
                time_till_capture = time.time() - start_time_color_change
                

                thread.join()  # Wait for the thread to complete
                if not results_container:
                    return None  # or handle this case as needed
                results = results_container[0]# Извлечение и вывод результатов
                # Assuming you have methods or logic to extract these from 'results'
                gaze_pitch = results.pitch[0]  # Convert numpy array to scalar
                gaze_yaw = results.yaw[0]      # Convert numpy array to scalar

                print(f"Gaze Pitch: {gaze_pitch}, Gaze Yaw: {gaze_yaw}\n")
        
            
            else:
                print("The cursor is outside the 20% area of the circle.")
            break

    cv2.imshow(window_name, np.zeros((monitor_pixels[1], monitor_pixels[0], 3), np.float32))
    cv2.waitKey(500)
    #source.clear_frame_buffer()
    return (            center, 
            time_till_capture, 
            quit_requested,
            gaze_pitch,
            gaze_yaw,
            paths_list
            )
