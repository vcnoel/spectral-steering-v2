import json
import matplotlib.pyplot as plt
import pandas as pd
import os

def generate_plots():
    if not os.path.exists("scaling_results.json"):
        print("Error: scaling_results.json not found.")
        return

    with open("scaling_results.json", "r") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    os.makedirs("results/plots", exist_ok=True)
    
    # 1. Plot Sycophancy vs Alpha per Model
    plt.figure(figsize=(10, 6))
    for model in df['model'].unique():
        m_df = df[(df['model'] == model) & (df['benchmark'] == 'sycophancy')]
        m_df = m_df.sort_values('alpha')
        plt.plot(m_df['alpha'], m_df['error_rate'], marker='o', label=model)
    
    plt.title("Sycophancy Error Rate vs Steering Alpha")
    plt.xlabel("Alpha (Negative = Sharpening, Positive = Smoothing)")
    plt.ylabel("Sycophancy Error Rate (%)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig("results/plots/sycophancy_scaling.png")
    plt.close()
    
    # 2. Plot GSM8K Accuracy vs Alpha (to show conservation)
    plt.figure(figsize=(10, 6))
    for model in df['model'].unique():
        m_df = df[(df['model'] == model) & (df['benchmark'] == 'gsm8k')]
        m_df = m_df.sort_values('alpha')
        # Map accuracies to % if they are in 0-1 range
        accs = [a if a > 1.0 else a*100 for a in m_df['error_rate']] # Error rate field reused for acc in scaling script
        plt.plot(m_df['alpha'], accs, marker='s', label=model)
        
    plt.title("GSM8K Accuracy over Steering Trajectory")
    plt.xlabel("Alpha")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig("results/plots/gsm8k_scaling.png")
    plt.close()

    # 3. SNR vs Optimal Alpha
    # Find alpha with lowest error for sycophancy for each model/layer
    best_alphas = []
    for model in df['model'].unique():
        for layer in df[df['model'] == model]['layer'].unique():
            sl = df[(df['model'] == model) & (df['layer'] == layer) & (df['benchmark'] == 'sycophancy')]
            if sl.empty: continue
            best_idx = sl['error_rate'].idxmin()
            best_alphas.append({
                "model": model,
                "snr": sl.loc[best_idx, 'snr'],
                "optimal_alpha": sl.loc[best_idx, 'alpha']
            })
    
    badf = pd.DataFrame(best_alphas)
    plt.figure(figsize=(8, 5))
    plt.scatter(badf['snr'], badf['optimal_alpha'], c='red', s=100, edgecolors='black')
    for i, row in badf.iterrows():
        plt.annotate(row['model'], (row['snr'], row['optimal_alpha']), xytext=(5,5), textcoords='offset points')
    
    plt.title("Spectral Profile Correlation: SNR vs Optimal Alpha")
    plt.xlabel("Signal-to-Noise Ratio (SNR)")
    plt.ylabel("Optimal Steering Alpha")
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.savefig("results/plots/snr_correlation.png")
    plt.close()

    print("Plots saved to results/plots/")

    # Generate Markdown Table
    table_md = "## Global Scaling Results Table\n\n| Model | Layer | SNR | Alpha | Sycophancy Error ↓ | GSM8K Acc ↑ |\n|-------|-------|-----|-------|--------------------|-------------|\n"
    
    # Pivot for clean display
    for model in df['model'].unique():
        for layer in df[df['model'] == model]['layer'].unique():
            for alpha in df['alpha'].unique():
                syco = df[(df['model'] == model) & (df['layer'] == layer) & (df['alpha'] == alpha) & (df['benchmark'] == 'sycophancy')]
                gsm = df[(df['model'] == model) & (df['layer'] == layer) & (df['alpha'] == alpha) & (df['benchmark'] == 'gsm8k')]
                
                if syco.empty: continue
                
                s_val = syco.iloc[0]['error_rate']
                g_val = gsm.iloc[0]['error_rate'] if not gsm.empty else 0.0
                snr_val = syco.iloc[0]['snr']
                
                table_md += f"| {model} | {layer} | {snr_val:.2f} | {alpha} | {s_val:.1f}% | {g_val:.1f}% |\n"
    
    with open("results/rebuttal_table.md", "w") as f:
        f.write(table_md)
    print("Table saved to results/rebuttal_table.md")

if __name__ == "__main__":
    generate_plots()
