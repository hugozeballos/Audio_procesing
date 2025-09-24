import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("metrics.csv")

# convertir columnas numéricas (ignora celdas vacías)
for col in ["stoi","lufs","dur_ref_s","dur_out_s"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# boxplot correcto de STOI por backend
df.boxplot(column="stoi", by="backend")
plt.title("STOI por backend")
plt.suptitle("")
plt.ylabel("STOI (0–1)")
plt.show()

# promedio de LUFS por backend
df.groupby("backend")["lufs"].mean().plot(kind="bar")
plt.title("LUFS promedio por backend")
plt.ylabel("LUFS (dB)")
plt.show()

# duración referencia vs salida
import seaborn as sns
sns.scatterplot(data=df, x="dur_ref_s", y="dur_out_s", hue="backend")
plt.title("Duración referencia vs salida")
plt.xlabel("Referencia (s)")
plt.ylabel("Salida (s)")
plt.show()
