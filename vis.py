from argparse import ArgumentParser
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import pyautogui
# Импортируем функции из файла visualization.py
from visualization import fix_qt_cv_mismatch, setup_figure, plot_screen, plot_target_on_screen, get_face_landmarks_in_ccs, plot_face_landmarks, plot_eye_to_target_on_screen_line, get_camera_matrix
import mediapipe as mp

def main(base_path: str, screen_height_mm_offset: int = 10):
    # Предварительная настройка
    fix_qt_cv_mismatch()
    camera_matrix, dist_coefficients = get_camera_matrix(base_path)
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
    parent_directory = os.path.dirname(base_path)
    # Считываем данные из CSV
    df = pd.read_csv(f'{parent_directory}/data.csv')
    
    # Список для хранения изображений графиков
    plot_images = []
    
    # Генерация графиков
    for idx, row in df.iterrows():
        #monitor_mm = tuple(map(int, row['monitor_mm'][1:-1].split(',')))
        #monitor_pixels = tuple(map(int, row['monitor_pixels'][1:-1].split(',')))
        #point_on_screen_px = tuple(map(int, row['point_on_screen'][1:-1].split(',')))
        monitor_mm = (400, 250)
        monitor_pixels = (1980, 1080)
        point_on_screen_px = (400, 600)

        fig, ax = setup_figure()
        plot_screen(ax, monitor_mm[0], monitor_mm[1], screen_height_mm_offset)
        point_on_screen_3d = plot_target_on_screen(ax, point_on_screen_px, monitor_mm, monitor_pixels, screen_height_mm_offset)

        #frame = cv2.imread(f'{base_path}/{row["file_name"]}')
        frame = cv2.imread(f'C:\\Users\\bokar\\Documents\\gaze_collection_stepa\\data\\p02\\day01\\2024_04_03-18_27_45-landmark.jpg')
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            face_model_all_transformed = get_face_landmarks_in_ccs(camera_matrix, dist_coefficients, frame.shape, results)
            plot_face_landmarks(ax, face_model_all_transformed)
            plot_eye_to_target_on_screen_line(ax, face_model_all_transformed, point_on_screen_3d)

        ax.view_init(-70, -90)
        plt.legend()
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        plot_image = cv2.imdecode(img_array, 1)
        plot_images.append(plot_image)
        plt.close(fig)  # Закрытие фигуры matplotlib для освобождения памяти

        

    # Создание окна OpenCV
    cv2.namedWindow('window', cv2.WINDOW_NORMAL)

    # Получение размеров экрана и установка размеров окна
    screen_width, screen_height = pyautogui.size()
    window_width, window_height = screen_width // 2, screen_height // 2
    window_x, window_y = screen_width // 4, screen_height // 4

    cv2.resizeWindow('window', window_width, window_height)
    cv2.moveWindow('window', window_x, window_y)

    # Индекс текущего изображения
    current_plot_index = 0

    # Карусель изображений
    while True:
        cv2.imshow('window', plot_images[current_plot_index])
        key = cv2.waitKey(0)
        
        if key == ord('q'):  # Выход из программы
            break
        elif key == ord('d'):  # Следующий график
            current_plot_index = (current_plot_index + 1) % len(plot_images)
        elif key == ord('a'):  # Предыдущий график
            current_plot_index = (current_plot_index - 1) % len(plot_images)
            
    cv2.destroyAllWindows()

# Вызов функции main с необходимыми параметрами
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, default='./data/p00')
    args = parser.parse_args()

    main(args.base_path)
