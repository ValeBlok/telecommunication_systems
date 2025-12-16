import numpy as np
import matplotlib.pyplot as plt

class CDMASystem:
    def __init__(self, n_stations=4, code_length=8):
        # Parameters:
        # n_stations: количество базовых станций
        # code_length: длина кодов Уолша
        self.n_stations = n_stations
        self.code_length = code_length
        
        self.messages = ["GOD", "CAT", "HAM", "SUN"]
        
        # Коды Уолша
        self.walsh_codes = self.generate_walsh_codes()
        # Кодировка слов в формат ASCII
        self.encoded_messages = self.encode_messages()
        # Распредеение кодов Уолша по станциям
        self.station_codes = {}
        for i in range(min(self.n_stations, len(self.walsh_codes))):
            self.station_codes[f"Station {chr(65+i)}"] = {
                'code': self.walsh_codes[i],
                'message': self.messages[i]
            }
    
    def generate_walsh_codes(self):
        # Матрица 1x1
        w = np.array([[1]])
        # Построение матрицы Уолша
        while w.shape[0] < self.code_length:
            n = w.shape[0]
            # Новая матрица Уолша как [W W; W -W]
            w_new = np.zeros((2*n, 2*n))
            w_new[:n, :n] = w
            w_new[:n, n:] = w
            w_new[n:, :n] = w
            w_new[n:, n:] = -w
            w = w_new
        
        # Первые n_stations строк
        codes = w[:min(self.n_stations, self.code_length)]
        return codes
    
    def encode_messages(self):
        # Кодирование сообщений в бинарную форму (ASCII)
        # Возвращает словарь закодированных сообщений
        encoded = {}
        for msg in self.messages:
            # Преобразование каждого символа в 8-битный ASCII код
            binary_msg = ''
            for char in msg:
                # ASCII код, преобразование в двоичную строку
                binary_char = format(ord(char), '08b')
                binary_msg += binary_char
            
            # Преобразование строки битов в массив чисел (+1/-1)
            encoded[msg] = np.array([1 if bit == '1' else -1 for bit in binary_msg])
        
        return encoded
    
    def modulate_signal(self, station_id):
        # Модуляция сигнала для конкретной станции
        # Принимаю station_id - идентификатор станции (A, B, C, D)
        # Возвращает модулированный сигнал
        station_key = f"Station {station_id}"
        if station_key not in self.station_codes:
            raise ValueError(f"Станция {station_id} не найдена")
        
        station = self.station_codes[station_key]
        code = station['code']
        message = station['message']
        encoded_msg = self.encoded_messages[message]
        
        # Распространение каждого бита сообщения на длину кода
        modulated = []
        for bit in encoded_msg:
            # Умножение бит на код Уолша
            modulated_chip = bit * code
            modulated.extend(modulated_chip)
        
        return np.array(modulated), station
    
    def transmit_signals(self):
        # Трансляция сигналов от всех станций и их смешивание в эфире
        # Возвращает смешанный сигнал и информацию о станциях
        all_signals = []
        station_info = {}
        
        print("=" * 60)
        print("ПЕРЕДАЧА СИГНАЛОВ ОТ БАЗОВЫХ СТАНЦИЙ")
        print("=" * 60)
        
        for station_id in ['A', 'B', 'C', 'D']:
            signal, station = self.modulate_signal(station_id)
            all_signals.append(signal)
            station_info[station_id] = station
            
            # Вывод информации о станции
            print(f"\nСтанция {station_id}:")
            print(f"  Сообщение: {station['message']}")
            print(f"  Код Уолша: {station['code']}")
            print(f"  Длина сигнала: {len(signal)} чипов")
        
        # Эмуляция беспроводной среды
        mixed_signal = np.sum(all_signals, axis=0)
        
        print("\n" + "=" * 60)
        print(f"СМЕШАННЫЙ СИГНАЛ В ЭФИРЕ (длина: {len(mixed_signal)} чипов)")
        print("=" * 60)
        print(f"Первые 20 чипов: {mixed_signal[:20]}")
        print(f"Последние 20 чипов: {mixed_signal[-20:]}")
        
        return mixed_signal, station_info, all_signals
    
    def demodulate_signal(self, mixed_signal, station_id, station_info):
        # Демодуляция сигнала для конкретной станции
        # Параметры:
        # mixed_signal - смешанный сигнал из эфира
        # station_id - идентификатор станции для демодуляции
        # station_info - информация о станциях
        # Возвращает демодулированное сообщение
        
        station_key = f"Station {station_id}"
        if station_key not in self.station_codes:
            raise ValueError(f"Станция {station_id} не найдена")
        
        code = station_info[station_id]['code']
        code_length = len(code)
        
        demodulated_bits = []
        
        # Разбиваем сигнал
        for i in range(0, len(mixed_signal), code_length):
            segment = mixed_signal[i:i+code_length]
            
            if len(segment) < code_length:
                break
            
            # Скалярное произведение с кодом Уолша
            correlation = np.dot(segment, code)
            # Нормализация и определение бит
            bit = 1 if correlation > 0 else -1 if correlation < 0 else 0
            demodulated_bits.append(bit)
        
        # Преобразуем биты (+1/-1) в строку (1/0)
        binary_str = ''.join(['1' if bit == 1 else '0' for bit in demodulated_bits])
        
        # Декодируем ASCII обратно в текст
        decoded_message = self.decode_binary_string(binary_str)
        
        return decoded_message, binary_str, demodulated_bits
    
    def decode_binary_string(self, binary_str):
        # Декодирование двоичной строки в текст
        # Параметр binary_str - двоичная строка
        # Возвращает декодированное сообщение

        # Разбиваем на 8-битные блоки
        chars = []
        for i in range(0, len(binary_str), 8):
            byte = binary_str[i:i+8]
            if len(byte) == 8:
                # Преобразуем двоичную строку в символ
                char_code = int(byte, 2)
                chars.append(chr(char_code))
        
        return ''.join(chars)
    
    def visualize_signals(self, all_signals, mixed_signal, station_info):
        # Визуализация сигналов
        # Параметры
        # all_signals - список сигналов от всех станций
        # mixed_signal - смешанный сигнал
        # station_info - информация о станциях
        fig, axes = plt.subplots(6, 1, figsize=(12, 10))
        
        # Отображение отдельных сигналов
        for i, (station_id, signal) in enumerate(zip(['A', 'B', 'C', 'D'], all_signals)):
            axes[i].plot(signal[:100], 'o-', markersize=3, linewidth=1)
            axes[i].set_title(f'Сигнал станции {station_id}: {station_info[station_id]["message"]}')
            axes[i].set_ylabel('Амплитуда')
            axes[i].grid(True, alpha=0.3)
        
        # Отображение смешанного сигнала
        axes[4].plot(mixed_signal[:100], 'o-', markersize=3, linewidth=1, color='red')
        axes[4].set_title('Смешанный сигнал в эфире')
        axes[4].set_ylabel('Амплитуда')
        axes[4].grid(True, alpha=0.3)
        
        # Отображение кодов Уолша
        codes_matrix = np.vstack([station_info[station_id]['code'] for station_id in ['A', 'B', 'C', 'D']])
        im = axes[5].imshow(codes_matrix, cmap='coolwarm', aspect='auto')
        axes[5].set_title('Коды Уолша для станций')
        axes[5].set_xlabel('Чипы')
        axes[5].set_ylabel('Станции')
        axes[5].set_yticks([0, 1, 2, 3])
        axes[5].set_yticklabels(['A', 'B', 'C', 'D'])
        plt.colorbar(im, ax=axes[5])
        
        plt.tight_layout()
        plt.show()

def main():
    # Основная функция демонстрации системы CDMA
    print("=" * 60)
    print("ПРОГРАММНАЯ МОДЕЛЬ СИСТЕМЫ CDMA")
    print("=" * 60)
    print("Конфигурация:")
    print("- Количество базовых станций: 4")
    print("- Длина кодов Уолша: 8")
    print("- Сообщения станций:")
    print("  * Станция A: 'GOD'")
    print("  * Станция B: 'CAT'")
    print("  * Станция C: 'HAM'")
    print("  * Станция D: 'SUN'")
    print("=" * 60)
    
    cdma = CDMASystem(n_stations=4, code_length=8)
    
    # Сгенерированные коды Уолша
    print("\nСГЕНЕРИРОВАННЫЕ КОДЫ УОЛША:")
    print("=" * 60)
    for i, code in enumerate(cdma.walsh_codes):
        print(f"Код {i} ({chr(65+i)}): {code}")
    
    # Трансляция сигналов
    mixed_signal, station_info, all_signals = cdma.transmit_signals()
    
    # Демодуляция сигналов для каждой станции
    print("\n" + "=" * 60)
    print("ДЕМОДУЛЯЦИЯ СИГНАЛОВ")
    print("=" * 60)
    
    for station_id in ['A', 'B', 'C', 'D']:
        decoded_message, binary_str, demod_bits = cdma.demodulate_signal(
            mixed_signal, station_id, station_info
        )
        
        print(f"\nДемодуляция для станции {station_id}:")
        print(f"  Ожидаемое сообщение: {station_info[station_id]['message']}")
        print(f"  Полученное сообщение: {decoded_message}")
        print(f"  Совпадение: {'ДА' if decoded_message == station_info[station_id]['message'] else 'НЕТ'}")
        print(f"  Первые 8 демодулированных бит: {demod_bits[:8]}")
    
    print("\n" + "=" * 60)
    print("=" * 60)
    
    # Визуализация
    cdma.visualize_signals(all_signals, mixed_signal, station_info)
    
    # Демонстрация ортогональности кодов Уолша
    print("\n" + "=" * 60)
    print("=" * 60)
    
    codes = [station_info[station_id]['code'] for station_id in ['A', 'B', 'C', 'D']]
    for i in range(len(codes)):
        for j in range(i, len(codes)):
            correlation = np.dot(codes[i], codes[j])
            print(f"Код {chr(65+i)} • Код {chr(65+j)} = {correlation}")

if __name__ == "__main__":
    main()
