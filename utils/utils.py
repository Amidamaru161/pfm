
import pandas as pd
import zipfile
import sys 
import os 
import time
import requests
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TICKER_LIST=['ABIO',
 'AFKS',
 'AFLT',
 'AKRN',
 'AMEZ',
 'APTK',
 'BLNG',
 'BSPB',
 'CHMF',
 'CHMK',
 'FEES',
 'FESH',
 'GAZP',
 'GCHE',
 'HYDR',
 'IRAO',
 'IRKT',
 'KMAZ',
 'LKOH',
 'LNZL',
 'LNZLP',
 'LSRG',
 'MAGN',
 'MGNT',
 'MRKC',
 'MRKK',
 'MRKP',
 'MRKU',
 'MRKV',
 'MRKY',
 'MRKZ',
 'MSNG',
 'MSRS',
 'MTLR',
 'MTSS',
 'MVID',
 'NKNC',
 'NKNCP',
 'NLMK',
 'NMTP',
 'NVTK',
 'OGKB',
 'PIKK',
 'PLZL',
 'RASP',
 'ROSN',
 'RTKM',
 'RTKMP',
 'SBER',
 'SBERP',
 'SNGS',
 'SNGSP',
 'SVAV',
 'TATN',
 'TATNP',
 'TGKA',
 'TGKB',
 'TRMK',
 'TRNFP',
 'VSMO',
 'VTBR']

INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]





def download_moex_candles(sec_ids, interval=60, limit=500, save_path='./data/', delay=0.01):
    """
    Загружает исторические данные свечей с MOEX для списка тикеров и сохраняет их в CSV.

    Параметры:
    sec_ids (list): Список тикеров.
    interval (int): Интервал свечей в минутах (по умолчанию 60).
    limit (int): Количество свечей за один запрос (по умолчанию 500).
    save_path (str): Директория для сохранения файлов (по умолчанию './data/').
    delay (float): Задержка между запросами в секундах (по умолчанию 0.01).
    """
    os.makedirs(save_path, exist_ok=True)

    for sec_id in sec_ids:
        url = f'https://iss.moex.com/iss/engines/stock/markets/shares/securities/{sec_id}/candles.json'
        params = {
            'start': '0',
            'interval': str(interval),
            'limit': str(limit),
        }
        print(f'START {sec_id}')
        start_time = time.time()
        df = pd.DataFrame()

        while True:
            try:
                response = requests.get(url=url, params=params)
                if response.status_code != 200:
                    print(f'Ошибка {response.status_code} для {sec_id}')
                    break
                data = response.json()
                if 'candles' not in data or not data['candles']['data']:
                    break
                temp_df = pd.DataFrame(data['candles']['data'], columns=data['candles']['columns'])
                df = pd.concat([df, temp_df])
                params['start'] = str(int(params['start']) + int(params['limit']))
                time.sleep(delay)
            except requests.exceptions.RequestException as e:
                print(f'Сетевая ошибка при загрузке {sec_id}: {e}')
                break
            except Exception as e:
                print(f'Ошибка при обработке данных {sec_id}: {e}')
                break

        if not df.empty:
            df['ticker'] = sec_id
            file_path = os.path.join(save_path, f'{sec_id}.csv')
            df.to_csv(file_path, index=False, encoding='UTF-8')
            end_time = time.time()
            print(f'Time spent = {round(end_time - start_time, 2)} s')
            print(f'File size = {os.stat(file_path).st_size} bytes')
        else:
            print(f'Нет данных для {sec_id}')

        print(f'END {sec_id}')


# Пример использования
# sec_ids = [
#     'ABIO', 'AFKS', 'AFLT', 'AKRN', 'AMEZ', 'APTK', 'BLNG', 'BSPB', 'CHMF', 'CHMK',
#     'ELFV', 'FEES', 'FESH', 'GAZP', 'GCHE', 'HYDR', 'INGR', 'IRAO', 'IRKT', 'KMAZ',
#     'LKOH', 'LNZLP', 'LNZL', 'LSRG', 'MAGN', 'MGNT', 'MRKC', 'MRKK', 'MRKP', 'MRKU',
#     'MRKV', 'MRKY', 'MRKZ', 'MSNG', 'MSRS', 'MTLR', 'MTSS', 'MVID', 'NKNCP', 'NKNC',
#     'NLMK', 'NMTP', 'NVTK', 'OGKB', 'PIKK', 'PLZL', 'RASP', 'ROSN', 'RTKMP', 'RTKM',
#     'SBERP', 'SBER', 'SNGSP', 'SNGS', 'SVAV', 'TATNP', 'TATN', 'TGKA', 'TGKB', 'TRMK',
#     'TRNFP', 'VSMO', 'VTBR'
# ]

# download_moex_candles(sec_ids)


def unzip_file(zip_path, extract_to='.'):
    """
    Разархивирует ZIP-архив в указанную директорию.
    
    :param zip_path: Путь к ZIP-архиву
    :param extract_to: Директория для извлечения (по умолчанию текущая)
    """
    try:
        # Создать директорию, если она не существует
        os.makedirs(extract_to, exist_ok=True)
        
        # Открыть ZIP-архив
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Извлечь все файлы
            zip_ref.extractall(extract_to)
            print(f'Архив успешно извлечен в: {extract_to}')
            
    except FileNotFoundError:
        print(f'Ошибка: файл {zip_path} не найден.')
    except zipfile.BadZipFile:
        print('Ошибка: файл не является ZIP-архивом или архив поврежден.')
    except Exception as e:
        print(f'Произошла ошибка: {str(e)}')

# Пример использования
#unzip_file('D1.zip', extract_to='D1_data')
#unzip_file('H1.zip', extract_to='H1_data')




def process_daily_data(input_folder, output_folder):
    """
    Обрабатывает CSV-файлы из указанной входной папки, преобразует данные в дневные интервалы
    и сохраняет результаты в выходную папку.
    
    Параметры:
    input_folder (str): Путь к папке с исходными CSV-файлами
    output_folder (str): Путь к папке для сохранения обработанных данных
    """
    # Создаем выходную папку, если она не существует
    os.makedirs(output_folder, exist_ok=True)
    
    # Обрабатываем каждый CSV-файл во входной папке
    for file in os.listdir(input_folder):
        if file.endswith(".csv"):
            # Читаем файл
            df = pd.read_csv(os.path.join(input_folder, file))
            
            # Преобразуем колонку 'begin' в datetime
            df['begin'] = pd.to_datetime(df['begin'])
            
            # Создаем колонку с датой (без времени)
            df['date'] = df['begin'].dt.date
            
            # Группируем по дате и агрегируем данные
            daily = df.groupby('date').agg({
                'open': 'first',
                'close': 'last',
                'high': 'max',
                'low': 'min',
                'value': 'sum',
                'volume': 'sum',
                'ticker': 'first'
            }).reset_index()
            
            # Сохраняем результат
            output_path = os.path.join(output_folder, file)
            daily.to_csv(output_path, index=False, encoding='utf-8')



def process_folder_data(folder_path,date_col='date', start_date='2010-01-01', threshold=0.01):
    """
    Обрабатывает все CSV-файлы в указанной папке, объединяет данные,
    фильтрует по дате и удаляет столбцы с пропусками.

    Параметры:
    - folder_path: путь к папке с CSV-файлами
    - start_date: начальная дата для фильтрации (по умолчанию '2010-01-01')
    - threshold: максимально допустимая доля пропусков (по умолчанию 0.01)

    Возвращает:
    - Объединенный и очищенный DataFrame
    """
    
    data_frames = []

    # Обработка каждого CSV-файла
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            
            # Загрузка и преобразование данных
            df = pd.read_csv(file_path)
            df['datetime'] = pd.to_datetime(df[date_col])
            df['tic'] = file_name.replace('.csv', '')
            
            # Преобразование в широкий формат
            df_pivot = df.pivot(index='datetime', columns='tic', values=['close','high','low','open','volume'])
            data_frames.append(df_pivot)

    # Объединение данных
    merged_data = pd.concat(data_frames, axis=1)
    
    # Фильтрация по дате
    merged_data = merged_data.loc[start_date:]
    
    # Удаление столбцов с пропусками
    null_percent = merged_data.isna().mean()
    columns_to_drop = null_percent[null_percent >= threshold].index
    merged_data = merged_data.drop(columns=columns_to_drop)

    return merged_data
