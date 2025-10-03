import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec

# 1) CSV de entrada (misma lógica que tu código)
candidates = [Path("dataset-audio-raw/enh/enh_metrics.csv"), Path("metrics.csv")]
csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else next((p for p in candidates if p.exists()), None)
if csv_path is None:
    raise SystemExit("No encuentro dataset-audio-raw/enh/enh_metrics.csv ni metrics.csv. Pasa la ruta como argumento.")

df = pd.read_csv(csv_path)

# 2) Normalización de tipos (igual que tu código)
numeric_cols = [
    "stoi","srmr","lufs_ref","lufs","delta_lufs",
    "peak_dbfs_ref","peak_dbfs_out","rms_dbfs_ref","rms_dbfs_out",
    "clip_rate_ref","clip_rate",
    "snr_db","si_sdr_db",
    "snr_db","si_sdr_db","snr_seg_db","spectral_dist_db",
    "dur_ref_s","dur_out_s","dur_diff_s"
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 3) Limpiezas mínimas opcionales
for c in ["dur_ref_s","dur_out_s"]:
    if c in df.columns:
        df.loc[df[c] <= 0, c] = pd.NA

# 4) NUEVA VISUALIZACIÓN: Dashboard Interconectado para Diarización
def create_diarization_dashboard():
    """Dashboard simplificado y funcional - sin sobreposiciones"""
    
    # Configuración general
    plt.style.use('default')
    sns.set_palette("husl")
    
    print("📈 Generando gráficos individuales (evitando sobreposiciones)...")
    
    # 1) SCATTER PLOT - Relación principal
    plt.figure(figsize=(10, 6))
    
    # Elegir las mejores métricas disponibles
    if 'snr_seg_db' in df.columns and 'stoi' in df.columns:
        x_metric, y_metric = 'snr_seg_db', 'stoi'
        x_label, y_label = 'SNR Segmental (dB)', 'STOI'
    elif 'si_sdr_db' in df.columns and 'stoi' in df.columns:
        x_metric, y_metric = 'si_sdr_db', 'stoi'
        x_label, y_label = 'SI-SDR (dB)', 'STOI'
    else:
        x_metric, y_metric = 'snr_db', 'stoi'
        x_label, y_label = 'SNR (dB)', 'STOI'
    
    if x_metric in df.columns and y_metric in df.columns:
        for backend in df['backend'].unique():
            subset = df[df['backend'] == backend]
            if len(subset) > 0:
                plt.scatter(subset[x_metric], subset[y_metric], 
                           label=backend, alpha=0.7, s=60)
        
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f'Relación {x_label} vs {y_label} - Diarización', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # 2) GRÁFICO DE BARRAS - Comparación de métricas principales
    plt.figure(figsize=(12, 6))
    
    # Métricas a comparar
    comparison_metrics = []
    if 'stoi' in df.columns: comparison_metrics.append('stoi')
    if 'si_sdr_db' in df.columns: comparison_metrics.append('si_sdr_db') 
    if 'snr_seg_db' in df.columns: comparison_metrics.append('snr_seg_db')
    if 'snr_db' in df.columns: comparison_metrics.append('snr_db')
    
    if comparison_metrics:
        # Calcular promedios por backend
        backend_means = df.groupby('backend')[comparison_metrics].mean()
        
        # Normalizar para comparación justa
        backend_normalized = backend_means.copy()
        for metric in comparison_metrics:
            if metric == 'stoi':
                backend_normalized[metric] = backend_means[metric]  # Ya está en 0-1
            else:
                # Normalizar métricas en dB a escala 0-1
                min_val = backend_means[metric].min()
                max_val = backend_means[metric].max()
                if max_val > min_val:
                    backend_normalized[metric] = (backend_means[metric] - min_val) / (max_val - min_val)
        
        x_pos = np.arange(len(backend_normalized))
        width = 0.8 / len(comparison_metrics)
        
        # Crear barras para cada métrica
        for i, metric in enumerate(comparison_metrics):
            offset = (i - len(comparison_metrics)/2) * width + width/2
            plt.bar(x_pos + offset, backend_normalized[metric], width, 
                   label=metric, alpha=0.8)
        
        plt.xlabel('Backend')
        plt.ylabel('Puntaje Normalizado (0-1)')
        plt.title('Comparación de Métricas por Backend (Normalizado)', fontweight='bold')
        plt.xticks(x_pos, backend_normalized.index, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
    
    # 3) HEATMAP de STOI por archivo (solo si hay pocos archivos)
    if 'stoi' in df.columns and 'rel' in df.columns:
        archivos_unicos = df['rel'].nunique()
        if archivos_unicos <= 10:  # Solo mostrar heatmap si hay pocos archivos
            plt.figure(figsize=(12, 6))
            
            heatmap_data = df.pivot_table(index='rel', columns='backend', values='stoi', aggfunc='mean')
            
            if not heatmap_data.empty:
                sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                           center=0.5, cbar_kws={'label': 'STOI'})
                plt.title('STOI por Archivo y Backend\n(Verde = mejor)', fontweight='bold')
                plt.tight_layout()
                plt.show()
        else:
            print(f"⚠️  Omitting heatmap (too many files: {archivos_unicos})")
    
    # 4) VIOLIN PLOT - Distribución de STOI
    if 'stoi' in df.columns:
        plt.figure(figsize=(10, 6))
        
        sns.violinplot(data=df, x='backend', y='stoi', inner='box')
        sns.swarmplot(data=df, x='backend', y='stoi', color='black', alpha=0.6, size=3)
        plt.title('Distribución de STOI por Backend', fontweight='bold')
        plt.ylabel('STOI')
        plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Excelente (>0.8)')
        plt.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Aceptable (>0.6)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    # 5) GRÁFICO DE LÍNEAS - Tendencia por archivo (solo si hay pocos)
    if 'stoi' in df.columns and 'rel' in df.columns and df['rel'].nunique() <= 8:
        plt.figure(figsize=(12, 6))
        
        # Ordenar archivos de manera consistente
        archivos_ordenados = sorted(df['rel'].unique())
        
        for backend in df['backend'].unique():
            valores = []
            for archivo in archivos_ordenados:
                valor = df[(df['backend'] == backend) & (df['rel'] == archivo)]['stoi'].mean()
                valores.append(valor if not np.isnan(valor) else None)
            
            # Filtrar valores None
            indices_validos = [i for i, v in enumerate(valores) if v is not None]
            valores_validos = [v for v in valores if v is not None]
            
            if valores_validos:
                plt.plot(indices_validos, valores_validos, 'o-', label=backend, markersize=6)
        
        plt.xlabel('Archivos (ordenados)')
        plt.ylabel('STOI')
        plt.title('Consistencia de STOI entre Archivos', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(range(len(archivos_ordenados)), [f"Archivo {i+1}" for i in range(len(archivos_ordenados))])
        plt.tight_layout()
        plt.show()

    print("✅ Dashboard generado exitosamente")

# 5) ANÁLISIS DE CORRELACIONES para diarización
def analyze_correlations():
    """Analizar cómo se correlacionan las métricas para diarización"""
    
    metrics_for_corr = ['stoi', 'si_sdr_db', 'snr_seg_db', 'snr_db', 'spectral_dist_db', 'srmr', 'lufs', 'clip_rate']
    available_corr = [m for m in metrics_for_corr if m in df.columns]
    
    if len(available_corr) >= 2:
        # Matriz de correlación
        corr_matrix = df[available_corr].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f', cbar_kws={'label': 'Correlación'})
        plt.title('Correlación entre Métricas para Diarización', fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Mostrar correlaciones con STOI (la más importante)
        if 'stoi' in corr_matrix.columns:
            stoi_correlations = corr_matrix['stoi'].sort_values(ascending=False)
            print("\n🔗 CORRELACIONES CON STOI (Inteligibilidad):")
            for metric, corr in stoi_correlations.items():
                if metric != 'stoi':
                    print(f"   {metric:12}: {corr:6.3f}")

    

# 6) RANKING FINAL para diarización
def final_ranking():
    """Ranking ponderado específico para diarización con nuevas métricas"""
    
    # Diferentes estrategias de ranking según métricas disponibles
    if 'stoi' in df.columns and 'si_sdr_db' in df.columns:
        # Estrategia 1: Énfasis en inteligibilidad (STOI + SI-SDR)
        if 'snr_seg_db' in df.columns:
            # Usar las 3 métricas más importantes
            df['puntaje_diarizacion'] = (
                0.5 * df['stoi'] + 
                0.3 * (df['si_sdr_db'].clip(lower=-20, upper=40) + 20) / 60 +
                0.2 * (df['snr_seg_db'].clip(lower=-10, upper=30) + 10) / 40
            )
        else:
            # Estrategia conservadora (solo STOI + SI-SDR)
            df['puntaje_diarizacion'] = (
                0.7 * df['stoi'] + 
                0.3 * (df['si_sdr_db'].clip(lower=-20, upper=40) + 20) / 60
            )
        
        ranking = df.groupby('backend')['puntaje_diarizacion'].agg(['mean', 'std', 'count']).round(4)
        ranking = ranking.sort_values('mean', ascending=False)
        
        print("\n🏆 RANKING FINAL PARA DIARIZACIÓN:")
        print("="*50)
        print(ranking)
        
        # Gráfico de ranking
        plt.figure(figsize=(12, 6))
        
        # Subgráfico 1: Ranking principal
        plt.subplot(1, 2, 1)
        bars = plt.bar(ranking.index, ranking['mean'], 
                      yerr=ranking['std'], capsize=5, alpha=0.7,
                      color=['#2ecc71' if x > 0.7 else '#f39c12' if x > 0.5 else '#e74c3c' 
                            for x in ranking['mean']])
        
        plt.title('Puntaje Combinado Diarización', fontweight='bold')
        plt.ylabel('Puntaje (0-1)')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        
        for bar, mean, std in zip(bars, ranking['mean'], ranking['std']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Subgráfico 2: Comparación de métricas individuales
        if 'snr_seg_db' in df.columns:
            plt.subplot(1, 2, 2)
            metric_means = df.groupby('backend')[['stoi', 'si_sdr_db', 'snr_seg_db']].mean()
            # Normalizar para comparación
            metric_means_norm = metric_means.copy()
            metric_means_norm['stoi'] = metric_means['stoi']  # Ya está en 0-1
            metric_means_norm['si_sdr_db'] = (metric_means['si_sdr_db'] + 20) / 60  # Normalizar a 0-1
            metric_means_norm['snr_seg_db'] = (metric_means['snr_seg_db'] + 10) / 40  # Normalizar a 0-1
            
            x_pos = np.arange(len(metric_means_norm))
            width = 0.25
            
            plt.bar(x_pos - width, metric_means_norm['stoi'], width, label='STOI', alpha=0.7)
            plt.bar(x_pos, metric_means_norm['si_sdr_db'], width, label='SI-SDR', alpha=0.7)
            plt.bar(x_pos + width, metric_means_norm['snr_seg_db'], width, label='SNR Seg', alpha=0.7)
            
            plt.title('Comparación de Métricas Individuales', fontweight='bold')
            plt.ylabel('Puntaje Normalizado (0-1)')
            plt.xticks(x_pos, metric_means_norm.index, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
# Ejecutar análisis completo
print("📊 INICIANDO ANÁLISIS AVANZADO PARA DIARIZACIÓN...")
print("="*60)

create_diarization_dashboard()
analyze_correlations() 
final_ranking()

# Mantener tu resumen tabular original
summary_cols = [c for c in ["stoi","srmr","snr_db","si_sdr_db","snr_seg_db","spectral_dist_db","lufs","delta_lufs","clip_rate"] if c in df.columns]
if summary_cols:
    summary = df.groupby("backend", dropna=True)[summary_cols].agg(["count","median","mean"])
    print("\n📋 RESUMEN TABULAR ORIGINAL:")
    print("="*50)
    print(summary)
    
    out_summary = csv_path.with_name("enh_metrics_summary_by_backend.csv")
    summary.to_csv(out_summary)
    print(f"\n💾 Resumen guardado en: {out_summary}")

print("\n✅ ANÁLISIS COMPLETADO")