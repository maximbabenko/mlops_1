import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
import os

def load_and_preprocess(input_path, output_path):
    """Загрузка и предварительная обработка данных о вине."""
    
    # Проверка существования входного пути
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path '{input_path}' does not exist.")

    # Загрузка данных
    red_wine = pd.read_csv(os.path.join(input_path, "winequality-red.csv"), sep=";")
    white_wine = pd.read_csv(os.path.join(input_path, "winequality-white.csv"), sep=";")

    # Объединение датасетов и добавление метки типа вина
    red_wine['type'] = 'red'
    white_wine['type'] = 'white'
    wines = pd.concat([red_wine, white_wine], ignore_index=True)

    # Очистка данных: удаление пропущенных значений
    wines.dropna(inplace=True)

    # Кодирование категориального признака (типа вина)
    wines['type'] = wines['type'].apply(lambda x: 1 if x == 'red' else 0)

    # Разделение признаков и целевого признака (качество вина)
    features = wines.drop(columns=['quality'])
    target = wines['quality']

    # Отделение категориального признака перед нормализацией
    wine_type = features['type']
    features_to_normalize = features.drop(columns=['type'])

    # Нормализация признаков
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features_to_normalize)

    # Преобразование нормализованных признаков в DataFrame и добавление 'type'
    processed_data = pd.DataFrame(features_normalized, columns=features_to_normalize.columns)
    processed_data['type'] = wine_type.reset_index(drop=True)

    # Добавление целевого признака 'quality'
    processed_data['quality'] = target.reset_index(drop=True)

    # Проверка существования выходного пути
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Сохранение очищенных и нормализованных данных
    processed_data.to_csv(os.path.join(output_path, "cleaned_normalized_wine_data.csv"), index=False)
    print(f"Processed data saved to '{output_path}/cleaned_normalized_wine_data.csv'.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_path> <output_path>")
        sys.exit(1)

    load_and_preprocess(sys.argv[1], sys.argv[2])
