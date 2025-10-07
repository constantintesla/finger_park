import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path


class FingerTappingAnalyzer:
    """
    Анализатор упражнения постукивания пальцами для оценки неврологических параметров.
    Оценивает: скорость, амплитуду, плавность движений, ритм, наличие замедлений.
    """
    
    def __init__(self, data_file: str):
        """
        Инициализация анализатора
        
        Args:
            data_file: Путь к CSV файлу с данными отслеживания
        """
        self.data = pd.read_csv(data_file)
        self.fps = self._estimate_fps()
        self.results = {}
        
    def _estimate_fps(self) -> float:
        """Оценка FPS видео на основе временных меток"""
        if len(self.data) < 2:
            return 30.0  # Значение по умолчанию
        
        time_diff = self.data['time'].diff().dropna()
        fps = 1.0 / time_diff.median()
        return fps
    
    def _get_hand_data(self, hand_idx: int) -> pd.DataFrame:
        """Получение данных для конкретной руки"""
        hand_columns = [col for col in self.data.columns if f'hand_{hand_idx}' in col]
        hand_data = self.data[['frame', 'time'] + hand_columns].copy()
        return hand_data
    
    def _calculate_finger_distance(self, hand_data: pd.DataFrame, hand_idx: int = 0) -> np.ndarray:
        """Вычисление расстояния между кончиками большого и указательного пальцев"""
        thumb_x = hand_data[f'hand_{hand_idx}_landmark_4_x']
        thumb_y = hand_data[f'hand_{hand_idx}_landmark_4_y']
        index_x = hand_data[f'hand_{hand_idx}_landmark_8_x']
        index_y = hand_data[f'hand_{hand_idx}_landmark_8_y']
        
        distances = np.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
        return distances
    
    def _detect_tapping_events(self, distances: np.ndarray, threshold: float = 20) -> List[int]:
        """
        Обнаружение событий постукивания на основе изменений расстояния между пальцами
        
        Args:
            distances: Массив расстояний между пальцами
            threshold: Пороговое значение для обнаружения постукивания
            
        Returns:
            Список индексов кадров с событиями постукивания
        """
        # Вычисляем производную (скорость изменения расстояния)
        distance_diff = np.diff(distances)
        
        # Находим пики в изменении расстояния
        peaks, _ = signal.find_peaks(np.abs(distance_diff), height=threshold)
        
        # Фильтруем события постукивания (сближение пальцев)
        tapping_events = []
        for peak in peaks:
            if distance_diff[peak] < -threshold:  # Сближение пальцев
                tapping_events.append(peak + 1)  # +1 из-за diff
        
        return tapping_events
    
    def analyze_speed(self, hand_data: pd.DataFrame, hand_idx: int = 0) -> Dict[str, float]:
        """
        Анализ скорости постукивания
        
        Returns:
            Словарь с метриками скорости
        """
        distances = self._calculate_finger_distance(hand_data, hand_idx)
        tapping_events = self._detect_tapping_events(distances)
        
        if len(tapping_events) < 2:
            return {
                'taps_per_second': 0.0,
                'average_interval': 0.0,
                'speed_consistency': 0.0
            }
        
        # Вычисляем интервалы между постукиваниями
        intervals = np.diff(tapping_events) / self.fps  # В секундах
        taps_per_second = len(tapping_events) / (len(hand_data) / self.fps)
        average_interval = np.mean(intervals)
        
        # Консистентность скорости (обратная величина от стандартного отклонения)
        speed_consistency = 1.0 / (np.std(intervals) + 1e-6)
        
        return {
            'taps_per_second': taps_per_second,
            'average_interval': average_interval,
            'speed_consistency': speed_consistency,
            'total_taps': len(tapping_events)
        }
    
    def analyze_amplitude(self, hand_data: pd.DataFrame, hand_idx: int = 0) -> Dict[str, float]:
        """
        Анализ амплитуды движений
        
        Returns:
            Словарь с метриками амплитуды
        """
        distances = self._calculate_finger_distance(hand_data, hand_idx)
        tapping_events = self._detect_tapping_events(distances)
        
        if len(tapping_events) < 2:
            return {
                'max_amplitude': 0.0,
                'min_amplitude': 0.0,
                'amplitude_range': 0.0,
                'amplitude_consistency': 0.0
            }
        
        # Амплитуда в моменты постукивания
        tapping_amplitudes = distances[tapping_events]
        
        max_amplitude = np.max(tapping_amplitudes)
        min_amplitude = np.min(tapping_amplitudes)
        amplitude_range = max_amplitude - min_amplitude
        
        # Консистентность амплитуды
        amplitude_consistency = 1.0 / (np.std(tapping_amplitudes) + 1e-6)
        
        return {
            'max_amplitude': max_amplitude,
            'min_amplitude': min_amplitude,
            'amplitude_range': amplitude_range,
            'amplitude_consistency': amplitude_consistency
        }
    
    def analyze_smoothness(self, hand_data: pd.DataFrame, hand_idx: int = 0) -> Dict[str, float]:
        """
        Анализ плавности движений
        
        Returns:
            Словарь с метриками плавности
        """
        distances = self._calculate_finger_distance(hand_data, hand_idx)
        
        # Вычисляем производные для анализа плавности
        first_derivative = np.diff(distances)
        second_derivative = np.diff(first_derivative)
        
        # Плавность как обратная величина от джерка (третья производная)
        jerk = np.diff(second_derivative)
        smoothness = 1.0 / (np.mean(np.abs(jerk)) + 1e-6)
        
        # Стабильность движения
        movement_stability = 1.0 / (np.std(first_derivative) + 1e-6)
        
        # Проверяем на nan
        if np.isnan(smoothness):
            smoothness = 0.0
        if np.isnan(movement_stability):
            movement_stability = 0.0
        if np.isnan(np.mean(np.abs(jerk))):
            jerk_magnitude = 0.0
        else:
            jerk_magnitude = np.mean(np.abs(jerk))
        
        return {
            'smoothness': smoothness,
            'movement_stability': movement_stability,
            'jerk_magnitude': jerk_magnitude
        }
    
    def analyze_rhythm(self, hand_data: pd.DataFrame, hand_idx: int = 0) -> Dict[str, float]:
        """
        Анализ ритмичности движений
        
        Returns:
            Словарь с метриками ритма
        """
        distances = self._calculate_finger_distance(hand_data, hand_idx)
        tapping_events = self._detect_tapping_events(distances)
        
        if len(tapping_events) < 3:
            return {
                'rhythm_consistency': 0.0,
                'rhythm_regularity': 0.0,
                'rhythm_score': 0.0
            }
        
        # Интервалы между постукиваниями
        intervals = np.diff(tapping_events) / self.fps
        
        # Консистентность ритма
        rhythm_consistency = 1.0 / (np.std(intervals) + 1e-6)
        
        # Регулярность ритма (корреляция с идеальным ритмом)
        ideal_intervals = np.full_like(intervals, np.mean(intervals))
        
        # Проверяем, не являются ли массивы постоянными
        if np.std(intervals) < 1e-6 or np.std(ideal_intervals) < 1e-6:
            # Если один из массивов постоянный, ритм идеальный
            rhythm_regularity = 1.0
        else:
            try:
                rhythm_regularity, _ = pearsonr(intervals, ideal_intervals)
                # Проверяем на nan
                if np.isnan(rhythm_regularity):
                    rhythm_regularity = 0.0
            except:
                # Если корреляция не может быть вычислена
                rhythm_regularity = 0.0
        
        # Общий балл ритма
        rhythm_score = (rhythm_consistency + abs(rhythm_regularity)) / 2
        
        # Проверяем на nan в итоговом балле
        if np.isnan(rhythm_score):
            rhythm_score = 0.0
        
        return {
            'rhythm_consistency': rhythm_consistency,
            'rhythm_regularity': rhythm_regularity,
            'rhythm_score': rhythm_score
        }
    
    def detect_decelerations(self, hand_data: pd.DataFrame, hand_idx: int = 0) -> Dict[str, any]:
        """
        Обнаружение замедлений и остановок
        
        Returns:
            Словарь с информацией о замедлениях
        """
        distances = self._calculate_finger_distance(hand_data, hand_idx)
        tapping_events = self._detect_tapping_events(distances)
        
        if len(tapping_events) < 3:
            return {
                'decelerations_detected': False,
                'deceleration_frames': [],
                'deceleration_severity': 0.0
            }
        
        # Анализ интервалов между постукиваниями
        intervals = np.diff(tapping_events) / self.fps
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # Замедления: интервалы значительно больше среднего
        deceleration_threshold = mean_interval + 2 * std_interval
        deceleration_events = np.where(intervals > deceleration_threshold)[0]
        
        # Степень замедления
        if len(deceleration_events) > 0:
            deceleration_severity = np.mean(intervals[deceleration_events]) / mean_interval
        else:
            deceleration_severity = 0.0
        
        # Безопасное получение кадров замедления
        deceleration_frames = []
        if len(deceleration_events) > 0 and len(tapping_events) > 0:
            try:
                deceleration_frames = tapping_events[deceleration_events].tolist()
            except (IndexError, TypeError):
                deceleration_frames = []
        
        return {
            'decelerations_detected': len(deceleration_events) > 0,
            'deceleration_frames': deceleration_frames,
            'deceleration_severity': deceleration_severity,
            'deceleration_count': len(deceleration_events)
        }
    
    def calculate_overall_score(self, metrics: Dict[str, float]) -> int:
        """
        Вычисление общего балла от 0 до 4 на основе всех метрик
        
        Args:
            metrics: Словарь с метриками анализа
            
        Returns:
            Балл от 0 до 4 (4 - наибольшая декомпенсация, 0 - минимальная)
        """
        # Нормализация метрик (0-1, где 1 = хороший результат)
        speed_score = min(metrics['taps_per_second'] / 5.0, 1.0)  # Норма: 5 постукиваний в секунду
        amplitude_score = min(metrics['amplitude_consistency'] / 10.0, 1.0)
        smoothness_score = min(metrics['smoothness'] / 100.0, 1.0)
        rhythm_score = metrics['rhythm_score']
        
        # Штрафы за замедления
        deceleration_penalty = max(0, metrics['deceleration_severity'] - 1.0) * 0.5
        
        # Общий балл (чем выше, тем лучше)
        overall_score = (speed_score + amplitude_score + smoothness_score + rhythm_score) / 4
        overall_score = max(0, overall_score - deceleration_penalty)
        
        # Преобразование в шкалу декомпенсации (4 - худший, 0 - лучший)
        # Инвертируем: 1 - overall_score, затем умножаем на 4
        decompensation_score = (1.0 - overall_score) * 4
        score_0_4 = int(round(decompensation_score))
        return min(4, max(0, score_0_4))
    
    def analyze_hand(self, hand_idx: int) -> Dict[str, any]:
        """
        Полный анализ одной руки
        
        Args:
            hand_idx: Индекс руки (0 или 1)
            
        Returns:
            Словарь с результатами анализа
        """
        hand_data = self._get_hand_data(hand_idx)
        
        if hand_data.empty:
            return {
                'hand_detected': False,
                'score': 0,
                'error': 'Рука не обнаружена'
            }
        
        # Анализ различных параметров
        speed_metrics = self.analyze_speed(hand_data, hand_idx)
        amplitude_metrics = self.analyze_amplitude(hand_data, hand_idx)
        smoothness_metrics = self.analyze_smoothness(hand_data, hand_idx)
        rhythm_metrics = self.analyze_rhythm(hand_data, hand_idx)
        deceleration_info = self.detect_decelerations(hand_data, hand_idx)
        
        # Объединение всех метрик
        all_metrics = {
            **speed_metrics,
            **amplitude_metrics,
            **smoothness_metrics,
            **rhythm_metrics,
            **deceleration_info
        }
        
        # Вычисление общего балла
        overall_score = self.calculate_overall_score(all_metrics)
        
        return {
            'hand_detected': True,
            'hand_side': 'правая' if hand_idx == 0 else 'левая',
            'score': overall_score,
            'metrics': all_metrics,
            'interpretation': self._interpret_score(overall_score)
        }
    
    def _interpret_score(self, score: int) -> str:
        """Интерпретация балла (4 - наибольшая декомпенсация, 0 - минимальная)"""
        interpretations = {
            4: "Наибольший уровень неврологической декомпенсации: выраженные нарушения, невозможность выполнить задание",
            3: "Высокий уровень декомпенсации: значительные нарушения, трудности с поддержанием ритма",
            2: "Умеренный уровень декомпенсации: заметные нарушения скорости или ритма",
            1: "Низкий уровень декомпенсации: незначительные нарушения, в целом удовлетворительное качество",
            0: "Минимальный уровень неврологической декомпенсации: отличное выполнение, высокая скорость, стабильная амплитуда, плавные движения, четкий ритм"
        }
        return interpretations.get(score, "Неопределенный результат")
    
    def analyze_both_hands(self) -> Dict[str, any]:
        """
        Анализ обеих рук
        
        Returns:
            Словарь с результатами анализа обеих рук
        """
        results = {}
        
        # Анализ каждой руки
        for hand_idx in [0, 1]:
            hand_key = f'hand_{hand_idx}'
            results[hand_key] = self.analyze_hand(hand_idx)
        
        # Сравнение рук
        if results['hand_0']['hand_detected'] and results['hand_1']['hand_detected']:
            score_diff = abs(results['hand_0']['score'] - results['hand_1']['score'])
            results['comparison'] = {
                'score_difference': score_diff,
                'dominant_hand': 'правая' if results['hand_0']['score'] > results['hand_1']['score'] else 'левая',
                'symmetry_score': 1.0 - (score_diff / 4.0)
            }
        else:
            results['comparison'] = {
                'score_difference': None,
                'dominant_hand': None,
                'symmetry_score': None
            }
        
        return results
    
    def generate_report(self, output_file: str = None) -> str:
        """
        Генерация отчета по анализу
        
        Args:
            output_file: Путь для сохранения отчета (опционально)
            
        Returns:
            Текст отчета
        """
        analysis_results = self.analyze_both_hands()
        
        report = "=== ОТЧЕТ ПО АНАЛИЗУ ПОСТУКИВАНИЯ ПАЛЬЦАМИ ===\n\n"
        
        # Анализ правой руки
        right_hand = analysis_results['hand_0']
        report += f"ПРАВАЯ РУКА:\n"
        if right_hand['hand_detected']:
            report += f"Балл: {right_hand['score']}/4\n"
            report += f"Интерпретация: {right_hand['interpretation']}\n"
            
            metrics = right_hand['metrics']
            report += f"Скорость: {metrics['taps_per_second']:.2f} постукиваний/сек\n"
            report += f"Амплитуда: {metrics['amplitude_range']:.2f} пикселей\n"
            report += f"Плавность: {metrics['smoothness']:.2f}\n"
            report += f"Ритм: {metrics['rhythm_score']:.2f}\n"
            if metrics['decelerations_detected']:
                report += f"Замедления: обнаружены ({metrics['deceleration_count']} раз)\n"
        else:
            report += "Рука не обнаружена\n"
        
        report += "\n"
        
        # Анализ левой руки
        left_hand = analysis_results['hand_1']
        report += f"ЛЕВАЯ РУКА:\n"
        if left_hand['hand_detected']:
            report += f"Балл: {left_hand['score']}/4\n"
            report += f"Интерпретация: {left_hand['interpretation']}\n"
            
            metrics = left_hand['metrics']
            report += f"Скорость: {metrics['taps_per_second']:.2f} постукиваний/сек\n"
            report += f"Амплитуда: {metrics['amplitude_range']:.2f} пикселей\n"
            report += f"Плавность: {metrics['smoothness']:.2f}\n"
            report += f"Ритм: {metrics['rhythm_score']:.2f}\n"
            if metrics['decelerations_detected']:
                report += f"Замедления: обнаружены ({metrics['deceleration_count']} раз)\n"
        else:
            report += "Рука не обнаружена\n"
        
        # Сравнение рук
        if analysis_results['comparison']['score_difference'] is not None:
            report += f"\nСРАВНЕНИЕ РУК:\n"
            report += f"Разница в баллах: {analysis_results['comparison']['score_difference']}\n"
            report += f"Доминирующая рука: {analysis_results['comparison']['dominant_hand']}\n"
            report += f"Симметрия: {analysis_results['comparison']['symmetry_score']:.2f}\n"
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Отчет сохранен в {output_file}")
        
        return report
    
    def save_detailed_results(self, output_file: str):
        """Сохранение детальных результатов в JSON"""
        analysis_results = self.analyze_both_hands()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        
        print(f"Детальные результаты сохранены в {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Анализ упражнения постукивания пальцами по 6.5.1.2")
    parser.add_argument("--input", required=True, help="CSV-файл с ключевыми точками")
    parser.add_argument("--output", help="JSON-отчёт (по умолчанию: <input>_analysis.json)")
    parser.add_argument("--report", help="Текстовый отчёт (по умолчанию: <input>_report.txt)")
    args = parser.parse_args()
    
    # Определение выходных файлов
    input_path = Path(args.input)
    if args.output:
        json_output = args.output
    else:
        json_output = input_path.with_name(input_path.stem + '_analysis.json')
    
    if args.report:
        report_output = args.report
    else:
        report_output = input_path.with_name(input_path.stem + '_report.txt')
    
    # Создание анализатора
    analyzer = FingerTappingAnalyzer(args.input)
    
    # Генерация отчета
    report = analyzer.generate_report(str(report_output))
    print(report)
    
    # Сохранение детальных результатов
    analyzer.save_detailed_results(str(json_output))


if __name__ == "__main__":
    main()
