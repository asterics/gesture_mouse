import pandas as pd
import matplotlib.pyplot as plt

csv_path = "../log.csv"
csv_df = pd.read_csv(csv_path, delimiter=";", usecols=list(range(1482)))
for i in range(18):
    normalized = csv_df[[f"ear_{i}",f"corrected_ear_{i}"]]/csv_df[[f"ear_{i}",f"corrected_ear_{i}"]].mean()
    normalized.plot.scatter(x=f"ear_{i}",y=f"corrected_ear_{i}")
    plt.show()
    print(normalized.describe())

