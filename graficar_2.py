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
    """Dashboard especializado para análisis de diarización"""
    
    # Configuración general
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Crear figura con GridSpec para layout más flexible
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)
    
    fig.suptitle('Dashboard de Análisis para Diarización de Audio', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # A) RADAR CHART - Evaluación multidimensional (arriba a la izquierda)
    ax_radar = fig.add_subplot(gs[0, 0], polar=True)
    
    # Métricas normalizadas para radar chart
    radar_metrics = ['stoi', 'si_sdr_db', 'snr_db', 'srmr']
    available_radar = [m for m in radar_metrics if m in df.columns]
    
    if available_radar:
        # Normalizar métricas (0-1)
        normalized_data = {}
        for backend in df['backend'].unique():
            backend_data = []
            for metric in available_radar:
                values = df[df['backend'] == backend][metric].dropna()
                if len(values) > 0:
                    if metric == 'stoi':  # STOI ya está en 0-1
                        norm_val = values.median()
                    else:  # Normalizar otras métricas
                        min_val = df[metric].min()
                        max_val = df[metric].max()
                        if max_val > min_val:
                            norm_val = (values.median() - min_val) / (max_val - min_val)
                        else:
                            norm_val = 0.5
                else:
                    norm_val = 0
                backend_data.append(norm_val)
            normalized_data[backend] = backend_data
        
        # Crear radar chart
        angles = np.linspace(0, 2*np.pi, len(available_radar), endpoint=False).tolist()
        angles += angles[:1]  # Cerrar el círculo
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(normalized_data)))
        
        for i, (backend, values) in enumerate(normalized_data.items()):
            values += values[:1]  # Cerrar el polígono
            ax_radar.plot(angles, values, 'o-', linewidth=2, label=backend, color=colors[i])
            ax_radar.fill(angles, values, alpha=0.1, color=colors[i])
        
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(available_radar)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Perfil Multidimensional\n(Área mayor = mejor)', fontweight='bold')
        ax_radar.legend(bbox_to_anchor=(1.1, 0.5), loc='center left')
    
    # B) SCATTER MATRIX - Correlaciones clave (arriba derecha)
    scatter_metrics = ['stoi', 'si_sdr_db', 'snr_db']
    available_scatter = [m for m in scatter_metrics if m in df.columns]
    
    if len(available_scatter) >= 2:
        ax_scatter = fig.add_subplot(gs[0, 1:3])
        
        # Scatter con colores por backend
        for backend in df['backend'].unique():
            subset = df[df['backend'] == backend]
            if len(subset) > 0:
                ax_scatter.scatter(subset[available_scatter[0]], 
                                 subset[available_scatter[1]], 
                                 label=backend, alpha=0.7, s=80)
        
        ax_scatter.set_xlabel(available_scatter[0])
        ax_scatter.set_ylabel(available_scatter[1])
        ax_scatter.set_title(f'Relación {available_scatter[0]} vs {available_scatter[1]}', 
                           fontweight='bold')
        ax_scatter.legend()
        ax_scatter.grid(True, alpha=0.3)
    
    # C) HEATMAP de ranking por archivo (centro izquierda)
    ax_heatmap = fig.add_subplot(gs[1, 0])
    
    if 'stoi' in df.columns and 'rel' in df.columns:
        # Crear matriz: archivos x backends
        heatmap_data = df.pivot_table(index='rel', columns='backend', values='stoi', aggfunc='mean')
        
        if not heatmap_data.empty:
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                       center=0.5, ax=ax_heatmap, cbar_kws={'label': 'STOI'})
            ax_heatmap.set_title('STOI por Archivo y Backend\n(Verde = mejor)', fontweight='bold')
            ax_heatmap.tick_params(axis='x', rotation=45)
            ax_heatmap.tick_params(axis='y', rotation=0)
    
    # D) BARRAS APILADAS - Distribución de calidad (centro derecha)
    ax_stacked = fig.add_subplot(gs[1, 1:3])
    
    if 'stoi' in df.columns:
        # Categorizar calidad
        def categorize_quality(stoi):
            if stoi > 0.8: return 'Excelente'
            elif stoi > 0.6: return 'Buena' 
            elif stoi > 0.4: return 'Regular'
            else: return 'Mala'
        
        df['calidad'] = df['stoi'].apply(categorize_quality)
        quality_counts = df.groupby(['backend', 'calidad']).size().unstack(fill_value=0)
        
        if not quality_counts.empty:
            quality_counts.plot(kind='bar', stacked=True, ax=ax_stacked, 
                              color=['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71'])
            ax_stacked.set_title('Distribución de Calidad por Backend', fontweight='bold')
            ax_stacked.set_ylabel('Número de Archivos')
            ax_stacked.legend(title='Calidad STOI')
            ax_stacked.tick_params(axis='x', rotation=45)
    
    # E) LÍNEAS TEMPORALES - Consistencia entre archivos (abajo izquierda)
    ax_lines = fig.add_subplot(gs[2, 0])
    
    if 'stoi' in df.columns and 'rel' in df.columns:
        # Ordenar archivos para mejor visualización
        archivos_ordenados = df['rel'].unique()
        
        for backend in df['backend'].unique():
            subset = df[df['backend'] == backend]
            if len(subset) > 0:
                valores = [subset[subset['rel'] == arch]['stoi'].mean() 
                          for arch in archivos_ordenados]
                ax_lines.plot(range(len(archivos_ordenados)), valores, 
                            'o-', label=backend, alpha=0.7, markersize=4)
        
        ax_lines.set_xlabel('Archivos (ordenados)')
        ax_lines.set_ylabel('STOI')
        ax_lines.set_title('Consistencia entre Archivos', fontweight='bold')
        ax_lines.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
        ax_lines.grid(True, alpha=0.3)
    
    # F) VIOLIN PLOT - Distribución detallada (abajo derecha)
    ax_violin = fig.add_subplot(gs[2, 1:3])
    
    if 'stoi' in df.columns:
        # Violin plot + swarm plot para ver puntos individuales
        sns.violinplot(data=df, x='backend', y='stoi', ax=ax_violin, inner='box')
        sns.swarmplot(data=df, x='backend', y='stoi', ax=ax_violin, 
                     color='black', alpha=0.6, size=3)
        ax_violin.set_title('Distribución Detallada de STOI', fontweight='bold')
        ax_violin.set_ylabel('STOI')
        ax_violin.tick_params(axis='x', rotation=45)
        ax_violin.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Umbral excelente')
        ax_violin.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Umbral aceptable')
        ax_violin.legend()
    
    plt.tight_layout()
    plt.show()

# 5) ANÁLISIS DE CORRELACIONES para diarización
def analyze_correlations():
    """Analizar cómo se correlacionan las métricas para diarización"""
    
    metrics_for_corr = ['stoi', 'si_sdr_db', 'snr_db', 'srmr', 'lufs', 'clip_rate']
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
    """Ranking ponderado específico para diarización"""
    
    if 'stoi' in df.columns and 'si_sdr_db' in df.columns:
        # Puntaje combinado: 70% STOI + 30% SI-SDR (énfasis en inteligibilidad)
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
        plt.figure(figsize=(10, 6))
        bars = plt.bar(ranking.index, ranking['mean'], 
                      yerr=ranking['std'], capsize=5, alpha=0.7,
                      color=['#2ecc71' if x > 0.7 else '#f39c12' if x > 0.5 else '#e74c3c' 
                            for x in ranking['mean']])
        
        plt.title('Puntaje Final para Diarización\n(70% STOI + 30% SI-SDR)', fontweight='bold')
        plt.ylabel('Puntaje (0-1)')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3, axis='y')
        
        for bar, mean, std in zip(bars, ranking['mean'], ranking['std']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{mean:.3f} ± {std:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()

# Ejecutar análisis completo
print("📊 INICIANDO ANÁLISIS AVANZADO PARA DIARIZACIÓN...")
print("="*60)

create_diarization_dashboard()
analyze_correlations() 
final_ranking()

# Mantener tu resumen tabular original
summary_cols = [c for c in ["stoi","srmr","snr_db","si_sdr_db","lufs","delta_lufs","clip_rate"] if c in df.columns]
if summary_cols:
    summary = df.groupby("backend", dropna=True)[summary_cols].agg(["count","median","mean"])
    print("\n📋 RESUMEN TABULAR ORIGINAL:")
    print("="*50)
    print(summary)
    
    out_summary = csv_path.with_name("enh_metrics_summary_by_backend.csv")
    summary.to_csv(out_summary)
    print(f"\n💾 Resumen guardado en: {out_summary}")

print("\n✅ ANÁLISIS COMPLETADO")