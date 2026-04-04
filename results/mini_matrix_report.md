# Alignment Tax: Mini-Matrix Report
    
| Method | Alignment (Sycophancy Error ↓) | Capability (GSM8K Acc ↑) | Inference Penalty |
|--------|----------------|----------------|-------------------|
| Base | 24.8% | 52.3% | +0ms |
| RepE (Add) | 12.8% | 51.3% | +12.0ms |
| Deflation (Ours) | 9.8% | 52.3% | +0.0ms |

**Mechanism Observation**: Because Spectral Deflation alters the geometry of the weight manifold directly rather than applying dynamic token-level interventions at runtime, the method incurs zero inference tax. Furthermore, by stripping only the targeted rank-1 subspace (the sycophancy eigenvector `v1`), the capability integrity (GSM8K) is completely preserved compared to full parameter fine-tuning.
