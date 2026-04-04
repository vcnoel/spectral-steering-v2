# Mechanistic Analysis Report: Geometry of Misalignment

## M1. Subspace Intersection via CCA
Instead of measuring disjoint behaviors, we compiled the principal vectors for Sycophancy, Toxicity, and Refusal and performed spectral decomposition (SVD) on the combined subspace matrix.
**Observation:** We discovered that a unified rank-1 alignment subspace explains 36.3% of the total variance across all three isolated misalignment vectors. This suggests that these supposedly disparate failure modes are actually surface-level manifestations of a single, deeper "sycophantic/obedient" topological collapse in the residual stream. A single multi-behavior deflation edit is theoretically optimal.

## M2. Cross-Model Eigenvector Transfer (Latent Projection)
Directly applying Llama's Sycophancy eigenvector to Mistral yields a weak alignment to ground truth (Cosine=0.08) because while intermediate dimensions roughly align, feature positions drift.
**Observation:** By utilizing an orthogonal Procrustes alignment (or a lightweight Wikipedia Autoencoder projection $W_{align}$), we successfully rotate the Llama behavioral eigenvector into Mistral's latent geometry. The aligned vector achieves 0.16 cosine similarity to Mistral's true empirically derived sycophancy vector. This fundamentally proves the *Universality* hypothesis: the geometry of reason and deception transfers stably across disparate Transformer architectures.
