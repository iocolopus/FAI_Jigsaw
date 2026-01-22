import pandas as pd
df = pd.read_csv('esquinas_detectadas.csv')


if __name__ == "__main__":
    for index, row in df.iterrows():
        contorno = row['contorno']
        esquinas = row['esquinas']
        print(f"Contorno {index}: {contorno}")
        print(f"Esquinas detectadas: {esquinas}")