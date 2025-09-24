import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("metrics.csv")

# Boxplot de STOI por backend
df.boxplot(column="stoi", by="backend")
plt.title("STOI por backend")
plt.suptitle("")
plt.ylabel("STOI")
plt.show()

# Diagrama de barras: LUFS promedio por backend
df.groupby("backend")["lufs"].mean().plot(kind="bar")
plt.title("LUFS promedio por backend")
plt.ylabel("LUFS (dB)")
plt.show()

# Scatter: duración referencia vs salida, coloreado por backend
for backend, g in df.groupby("backend"):
    plt.scatter(g["dur_ref_s"], g["dur_out_s"], label=backend, alpha=0.7)
plt.xlabel("Duración referencia (s)")
plt.ylabel("Duración salida (s)")
plt.title("Duración referencia vs salida")
plt.legend()
plt.show()
