import argparse
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
from filterpy.kalman import KalmanFilter
import mediapipe as mp


class FingerSmoother:
    def __init__(self, num_landmarks=21, dt=1.0, process_noise=1e-5, measurement_noise=1e-4):
        self.num_landmarks = num_landmarks
        self.kalman_filters = []
        
        # Создаем отдельный фильтр Калмана для каждой точки руки
        for _ in range(num_landmarks):
            kf = KalmanFilter(dim_x=4, dim_z=2)  # x, y, vx, vy
            kf.F = np.array([[1, 0, dt, 0],
                            [0, 1, 0, dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])  # Матрица состояния
            
            kf.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])  # Матрица измерения
            
            kf.R = np.eye(2) * measurement_noise  # Шум измерения
            kf.Q = np.eye(4) * process_noise  # Шум процесса
            
            kf.P = np.eye(4) * 10  # Ковариационная матрица
            
            self.kalman_filters.append(kf)
    
    def update(self, landmarks):
        smoothed_landmarks = np.zeros_like(landmarks)
        
        for i, (x, y) in enumerate(landmarks):
            if i >= self.num_landmarks:
                break
                
            kf = self.kalman_filters[i]
            
            # Предсказание
            kf.predict()
            
            # Обновление на основе измерения
            if x > 0 and y > 0:  # Если точка обнаружена
                measurement = np.array([x, y])
                kf.update(measurement)
                smoothed_landmarks[i] = kf.x[:2].flatten()
            else:
                # Если точка не обнаружена, используем предсказанное значение
                smoothed_landmarks[i] = kf.x[:2].flatten()
        
        return smoothed_landmarks


class FingerTracker:
    def __init__(self, smoothing=True):
        # Инициализация MediaPipe для отслеживания рук
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Инициализация сглаживателя
        if smoothing:
            self.smoother = FingerSmoother()
            print("Используется фильтр Калмана для сглаживания")
        else:
            self.smoother = None
            print("Сглаживание отключено")
    
    def get_hand_landmarks(self, frame):
        """Получение ключевых точек рук из кадра"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        landmarks_data = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Извлекаем координаты 21 точки руки
                hand_points = []
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    hand_points.append([x, y])
                
                landmarks_data.append(np.array(hand_points))
        
        return landmarks_data
    
    def detect_tapping_motion(self, landmarks_history, threshold=20):
        """Обнаружение постукивающих движений между указательным и большим пальцами"""
        if len(landmarks_history) < 3:
            return False, 0
        
        # Индексы точек: 4 - кончик большого пальца, 8 - кончик указательного пальца
        thumb_tip = landmarks_history[-1][4]
        index_tip = landmarks_history[-1][8]
        
        # Вычисляем расстояние между кончиками пальцев
        distance = np.linalg.norm(thumb_tip - index_tip)
        
        # Проверяем, есть ли движение (изменение расстояния)
        if len(landmarks_history) >= 2:
            prev_distance = np.linalg.norm(landmarks_history[-2][4] - landmarks_history[-2][8])
            distance_change = abs(distance - prev_distance)
            
            return distance_change > threshold, distance
        
        return False, distance


def process_video(input_path, output_path, show=False, smoothing=True):
    # Инициализация трекера пальцев
    tracker = FingerTracker(smoothing=smoothing)
    
    # Открытие видеофайла
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео {input_path}")
        return
    
    # Получение информации о видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Информация о видео: {width}x{height}, {fps:.2f} fps, {total_frames} кадров")
    
    # Подготовка для сохранения результатов
    all_landmarks = []
    frame_numbers = []
    landmarks_history = []
    
    # Обработка видео
    for frame_num in tqdm(range(total_frames), desc="Обработка видео"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Получение ключевых точек рук
        landmarks_list = tracker.get_hand_landmarks(frame)
        
        frame_data = {
            'frame': frame_num,
            'time': frame_num / fps,
            'hands_detected': len(landmarks_list)
        }
        
        # Обработка каждой обнаруженной руки
        for hand_idx, landmarks in enumerate(landmarks_list):
            # Сглаживание ключевых точек
            if tracker.smoother is not None:
                landmarks = tracker.smoother.update(landmarks)
            
            # Сохранение координат каждой точки руки
            for i, (x, y) in enumerate(landmarks):
                frame_data[f'hand_{hand_idx}_landmark_{i}_x'] = float(x)
                frame_data[f'hand_{hand_idx}_landmark_{i}_y'] = float(y)
            
            # Добавление в историю для анализа движения
            landmarks_history.append(landmarks)
            if len(landmarks_history) > 10:  # Сохраняем только последние 10 кадров
                landmarks_history.pop(0)
            
            # Обнаружение постукивающих движений
            is_tapping, finger_distance = tracker.detect_tapping_motion(landmarks_history)
            frame_data[f'hand_{hand_idx}_is_tapping'] = is_tapping
            frame_data[f'hand_{hand_idx}_finger_distance'] = float(finger_distance)
        
        all_landmarks.append(frame_data)
        
        # Визуализация
        if show:
            annotated_frame = frame.copy()
            
            # Рисуем ключевые точки рук
            for hand_idx, landmarks in enumerate(landmarks_list):
                if tracker.smoother is not None:
                    landmarks = tracker.smoother.update(landmarks)
                
                # Рисуем точки
                for i, (x, y) in enumerate(landmarks):
                    if x > 0 and y > 0:
                        color = (0, 255, 0) if i in [4, 8] else (255, 0, 0)  # Выделяем кончики пальцев
                        cv2.circle(annotated_frame, (int(x), int(y)), 3, color, -1)
                
                # Рисуем соединения между точками руки
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),  # Большой палец
                    (0, 5), (5, 6), (6, 7), (7, 8),  # Указательный палец
                    (0, 9), (9, 10), (10, 11), (11, 12),  # Средний палец
                    (0, 13), (13, 14), (14, 15), (15, 16),  # Безымянный палец
                    (0, 17), (17, 18), (18, 19), (19, 20)  # Мизинец
                ]
                
                for start, end in connections:
                    if (start < len(landmarks) and end < len(landmarks) and 
                        landmarks[start][0] > 0 and landmarks[start][1] > 0 and
                        landmarks[end][0] > 0 and landmarks[end][1] > 0):
                        start_point = (int(landmarks[start][0]), int(landmarks[start][1]))
                        end_point = (int(landmarks[end][0]), int(landmarks[end][1]))
                        cv2.line(annotated_frame, start_point, end_point, (0, 255, 0), 1)
                
                # Выделяем кончики большого и указательного пальцев
                if len(landmarks) > 8:
                    thumb_tip = (int(landmarks[4][0]), int(landmarks[4][1]))
                    index_tip = (int(landmarks[8][0]), int(landmarks[8][1]))
                    cv2.circle(annotated_frame, thumb_tip, 8, (0, 0, 255), -1)
                    cv2.circle(annotated_frame, index_tip, 8, (0, 0, 255), -1)
            
            # Добавляем информацию о кадре
            cv2.putText(annotated_frame, f"Кадр: {frame_num}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Отслеживание пальцев', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Закрытие видео
    cap.release()
    if show:
        cv2.destroyAllWindows()
    
    # Сохранение результатов в CSV
    if all_landmarks:
        df = pd.DataFrame(all_landmarks)
        df.to_csv(output_path, index=False)
        print(f"Результаты сохранены в {output_path}")
        print(f"Всего обработано кадров: {len(all_landmarks)}")
    else:
        print("Ключевые точки рук не обнаружены в видео.")


def main():
    parser = argparse.ArgumentParser(
        description="Отслеживание движений пальцев для анализа постукивания",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input',
        required=True,
        help="Путь к видеофайлу для анализа"
    )
    parser.add_argument(
        '--output',
        default="finger_tracking_results.csv",
        help="Файл для сохранения результатов"
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help="Показать анализ в реальном времени"
    )
    parser.add_argument(
        '--no-smoothing',
        action='store_true',
        help="Отключить сглаживание движения"
    )
    args = parser.parse_args()
    
    process_video(args.input, args.output, args.show, not args.no_smoothing)


if __name__ == "__main__":
    main()
