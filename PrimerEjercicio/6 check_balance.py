import pandas as pd

# Cargar datos
df = pd.read_csv("Bank_churn_modelling.csv", sep=";", encoding="utf-8")

# Verificar distribución de la variable objetivo
print("=" * 50)
print("ANÁLISIS DE BALANCE DEL DATASET")
print("=" * 50)

# Distribución de la variable objetivo
target_dist = df['Exited'].value_counts()
total = len(df)

print(f"\nDistribución de la variable objetivo:")
print(f"Deserción (1): {target_dist[1]} clientes ({target_dist[1]/total*100:.1f}%)")
print(f"No Deserción (0): {target_dist[0]} clientes ({target_dist[0]/total*100:.1f}%)")

#F1-Score bajo indica problemas de balance
# Calcular ratio de balance
balance_ratio = min(target_dist) / max(target_dist)
print(f"\nRatio de balance: {balance_ratio:.3f}")


# Interpretación del balance
if balance_ratio >= 0.8:
    print("✓ Dataset BALANCEADO (ratio >= 0.8)")
elif balance_ratio >= 0.5:
    print("⚠ Dataset MODERADAMENTE DESBALANCEADO (0.5 <= ratio < 0.8)")
elif balance_ratio >= 0.2:
    print("⚠ Dataset DESBALANCEADO (0.2 <= ratio < 0.5)")
else:
    print("✗ Dataset MUY DESBALANCEADO (ratio < 0.2)")

print(f"\nTotal de clientes: {total}")
print("=" * 50) 