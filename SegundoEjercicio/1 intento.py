import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Carga del archivo CSV")

try:
    
    with open("fri.csv", 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    print(f"Archivo leído: {len(lines)} líneas")
    
    # Procesar encabezado
    header_line = lines[0].strip()
    header = header_line.replace('"', '').split(',')
    print(f"Encabezado: {header}")
    
    # Procesar datos
    data = []
    for i, line in enumerate(lines[1:], 1):
        line = line.strip()
        if line:  # Si la línea no está vacía
            values = line.split(',')
            if len(values) == 3:
                try:
                    freq = float(values[0])
                    trans = float(values[1])
                    ing = float(values[2])
                    data.append([freq, trans, ing])
                except ValueError:
                    print(f"Error en línea {i}: {line}")
                    continue
    
    print(f"Datos procesados: {len(data)} filas")
    
    # Crear DataFrame
    df = pd.DataFrame(data, columns=header)
    
    print(" Archivo cargado exitosamente!")
    print(f"Dataset: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"Columnas: {list(df.columns)}")
    print("\nPrimeras 5 filas:")
    print(df.head())
    
    # Estadísticas básicas
    print("\n Estadísticas descriptivas:")
    print(df.describe())
    
    # Verificar valores nulos
    print("\n Valores nulos:")
    print(df.isnull().sum())
    
    # Generar gráficos
    print("\nGenerando gráficos...")
    
    # Histogramas
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(df['Frecuencia'], bins=20, ax=axes[0], color='skyblue')
    axes[0].set_title('Distribución - Frecuencia')
    sns.histplot(df['Transaccionalidad'], bins=30, ax=axes[1], color='orange')
    axes[1].set_title('Distribución - Transaccionalidad')
    sns.histplot(df['Ingresos'], bins=30, ax=axes[2], color='green')
    axes[2].set_title('Distribución - Ingresos')
    plt.tight_layout()
    plt.savefig('histogramas_distribucion.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Boxplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.boxplot(y=df['Frecuencia'], ax=axes[0], color='skyblue')
    axes[0].set_title('Boxplot - Frecuencia')
    sns.boxplot(y=df['Transaccionalidad'], ax=axes[1], color='orange')
    axes[1].set_title('Boxplot - Transaccionalidad')
    sns.boxplot(y=df['Ingresos'], ax=axes[2], color='green')
    axes[2].set_title('Boxplot - Ingresos')
    plt.tight_layout()
    plt.savefig('boxplots_outliers.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Matriz de correlación
    corr = df.corr()
    plt.figure(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de Correlación')
    plt.savefig('matriz_correlacion_fri.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n Gráficos guardados exitosamente:")
    print("   - histogramas_distribucion.png")
    print("   - boxplots_outliers.png")
    print("   - matriz_correlacion_fri.png")
    
    print("\n RESUMEN FINAL:")
    print(f"   - Dataset: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"   - Variables: {list(df.columns)}")
    print(f"   - Rango de valores:")
    print(f"     * Frecuencia: {df['Frecuencia'].min():.0f} - {df['Frecuencia'].max():.0f}")
    print(f"     * Transaccionalidad: {df['Transaccionalidad'].min():.2f} - {df['Transaccionalidad'].max():.2f}")
    print(f"     * Ingresos: {df['Ingresos'].min():.2f} - {df['Ingresos'].max():.2f}")
    
    print("\n ANÁLISIS EXITOSO!")

except Exception as e:
    print(f" Error: {e}")
    print("Detalles del error:")
    import traceback
    traceback.print_exc()
