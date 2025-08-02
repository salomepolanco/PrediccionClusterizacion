import pandas as pd

# Cargar resultados
df = pd.read_csv('importancia_variables.csv')

print("=" * 60)
print("RESULTADOS DE SELECCIÓN DE VARIABLES")
print("=" * 60)

print("\nTODAS LAS VARIABLES ORDENADAS POR IMPORTANCIA:")
print("-" * 50)
for i, row in df.iterrows():
    print(f"{i+1:2d}. {row['Variable']:15s} - Puntuación: {row['Composite_Score']:.3f}")

print(f"\n" + "=" * 40)
print("RECOMENDACIÓN FINAL")
print("=" * 40)

# Variables de alta importancia (>0.5)
high_importance = df[df['Composite_Score'] > 0.5]
print(f"\n🔴 VARIABLES DE ALTA IMPORTANCIA (Puntuación > 0.5):")
print("-" * 50)
for i, row in high_importance.iterrows():
    print(f"{i+1}. {row['Variable']} (Puntuación: {row['Composite_Score']:.3f})")

# Variables de media importancia (0.2-0.5)
medium_importance = df[(df['Composite_Score'] > 0.2) & (df['Composite_Score'] <= 0.5)]
print(f"\n🟡 VARIABLES DE MEDIA IMPORTANCIA (0.2 < Puntuación ≤ 0.5):")
print("-" * 50)
for i, row in medium_importance.iterrows():
    print(f"{i+1}. {row['Variable']} (Puntuación: {row['Composite_Score']:.3f})")

# Variables recomendadas para el modelo
recommended_vars = df[df['Composite_Score'] > 0.2]['Variable'].tolist()
print(f"\n✅ VARIABLES RECOMENDADAS PARA EL MODELO (Puntuación > 0.2):")
print("-" * 50)
for i, var in enumerate(recommended_vars, 1):
    score = df[df['Variable'] == var]['Composite_Score'].iloc[0]
    print(f"{i}. {var} (Puntuación: {score:.3f})")

print(f"\n📊 RESUMEN:")
print(f"   - Total de variables: {len(df)}")
print(f"   - Variables recomendadas: {len(recommended_vars)}")
print(f"   - Variables excluidas: {len(df) - len(recommended_vars)}")

print("\n" + "=" * 60) 