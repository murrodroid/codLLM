# Paired publication intervals

10,000 paired connected-COD/source-record bayesian bootstrap replicates; 95% percentile intervals. Differences are first minus second, in percentage points.

Conditional on the selected models and observed archive panel. Validation selection and training-seed variation are not covered; intervals are exploratory and not multiplicity-adjusted.

The published metrics are recomputed in every draw (pooled macro: reference ∪ predictions per model; source reference macro: reference labels per archive). Positive Bayesian weights retain observed support. Ordinary multinomial draws may omit rare labels; fixed-support sensitivity metrics assign absent labels zero F1. The CSV/JSON retain both definitions and rare-label diagnostics.

Paired Bayesian connected COD/source-record cluster bootstrap, Dirichlet(1,...,1) group weights; 95% equal-tail credible intervals on the observed empirical support.

Bayesian draws assign normalized independent Exp(1) weights to connected groups, equivalent to Dirichlet(1,...,1) weights. Every model receives exactly the same group weights in each draw. We apply [Rubin's Bayesian bootstrap](https://people.eecs.berkeley.edu/~jordan/sail/readings/rubin.pdf) to the connected COD/source-record units, retaining row-frequency and equal-source metric definitions.

The central interval of bootstrap metric draws need not contain the original point estimate: F1 is nonlinear and sparse support can shift its resampling distribution. Bayesian weighting does not supply missing evidence within labels represented by a single group. Treat these intervals as conditional exploratory evidence, not calibrated uncertainty over unseen label cases or new archives.

## Matched comparisons

| Change | First | Second | Pooled macro F1 Δ [95% interval] | Micro F1 Δ [95% interval] | Equal-source ref. macro F1 Δ [95% interval] |
|---|---|---|---:|---:|---:|
| pretraining | cod-p10/floor-0_synthetic-0.3_pretrain-48 | cod-p10/floor-0_synthetic-0.3_pretrain-4 | -0.011 [-0.483, +0.427] | -1.403 [-3.599, -0.156] | -0.154 [-0.600, +0.283] |
| synthesis | cod-p10/floor-0_synthetic-0.6_pretrain-4 | cod-p10/floor-0_synthetic-0.3_pretrain-4 | -0.849 [-1.214, -0.364] | -1.791 [-4.388, -0.525] | -0.684 [-0.994, -0.171] |
| floor | cod-p10/floor-0_synthetic-0.3_pretrain-4 | cod-p10/floor-450_synthetic-0.3_pretrain-4 | +1.544 [+1.144, +2.089] | +0.917 [-0.106, +2.318] | +0.257 [-0.207, +0.798] |
| metadata | metadata-p10/floor-0_synthetic-0.3_pretrain-4 | cod-p10/floor-0_synthetic-0.3_pretrain-4 | -1.008 [-1.507, -0.558] | -1.403 [-3.998, +0.049] | -0.656 [-1.058, -0.223] |
| patience | cod-p20/floor-0_synthetic-0.3_pretrain-4 | cod-p10/floor-0_synthetic-0.3_pretrain-4 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| synthesis | cod-p10/floor-0_synthetic-0.6_pretrain-48 | cod-p10/floor-0_synthetic-0.3_pretrain-48 | -0.174 [-0.548, +0.316] | -0.026 [-0.486, +0.544] | +1.055 [+0.683, +1.476] |
| floor | cod-p10/floor-0_synthetic-0.3_pretrain-48 | cod-p10/floor-450_synthetic-0.3_pretrain-48 | +2.394 [+1.958, +2.892] | +7.067 [+0.348, +21.162] | +1.400 [+0.960, +1.850] |
| metadata | metadata-p10/floor-0_synthetic-0.3_pretrain-48 | cod-p10/floor-0_synthetic-0.3_pretrain-48 | +0.103 [-0.330, +0.626] | -0.320 [-2.904, +1.905] | +0.732 [+0.240, +1.122] |
| patience | cod-p20/floor-0_synthetic-0.3_pretrain-48 | cod-p10/floor-0_synthetic-0.3_pretrain-48 | +0.390 [+0.099, +0.700] | -0.385 [-1.878, +0.550] | -0.006 [-0.324, +0.238] |
| pretraining | cod-p10/floor-0_synthetic-0.6_pretrain-48 | cod-p10/floor-0_synthetic-0.6_pretrain-4 | +0.664 [+0.176, +1.119] | +0.363 [-1.869, +3.017] | +1.586 [+1.091, +1.927] |
| floor | cod-p10/floor-0_synthetic-0.6_pretrain-4 | cod-p10/floor-450_synthetic-0.6_pretrain-4 | +0.250 [-0.235, +0.690] | +0.420 [-0.056, +1.164] | -0.947 [-1.271, -0.455] |
| metadata | metadata-p10/floor-0_synthetic-0.6_pretrain-4 | cod-p10/floor-0_synthetic-0.6_pretrain-4 | -0.601 [-1.044, -0.171] | +0.072 [-0.297, +0.494] | +0.226 [-0.219, +0.632] |
| patience | cod-p20/floor-0_synthetic-0.6_pretrain-4 | cod-p10/floor-0_synthetic-0.6_pretrain-4 | +0.992 [+0.604, +1.290] | +0.338 [+0.142, +0.614] | +0.162 [-0.219, +0.383] |
| floor | cod-p10/floor-0_synthetic-0.6_pretrain-48 | cod-p10/floor-450_synthetic-0.6_pretrain-48 | +0.655 [+0.275, +1.144] | +0.881 [-0.484, +3.251] | +0.272 [-0.101, +0.595] |
| metadata | metadata-p10/floor-0_synthetic-0.6_pretrain-48 | cod-p10/floor-0_synthetic-0.6_pretrain-48 | -0.040 [-0.538, +0.433] | -2.726 [-8.124, +0.476] | +0.440 [-0.025, +0.803] |
| patience | cod-p20/floor-0_synthetic-0.6_pretrain-48 | cod-p10/floor-0_synthetic-0.6_pretrain-48 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| pretraining | cod-p10/floor-450_synthetic-0.3_pretrain-48 | cod-p10/floor-450_synthetic-0.3_pretrain-4 | -0.860 [-1.296, -0.361] | -7.553 [-21.645, -0.660] | -1.296 [-1.751, -0.774] |
| synthesis | cod-p10/floor-450_synthetic-0.6_pretrain-4 | cod-p10/floor-450_synthetic-0.3_pretrain-4 | +0.446 [+0.157, +1.028] | -1.294 [-3.780, -0.027] | +0.519 [+0.115, +1.028] |
| metadata | metadata-p10/floor-450_synthetic-0.3_pretrain-4 | cod-p10/floor-450_synthetic-0.3_pretrain-4 | -0.753 [-1.176, -0.194] | -1.604 [-4.665, +0.106] | -0.994 [-1.433, -0.472] |
| patience | cod-p20/floor-450_synthetic-0.3_pretrain-4 | cod-p10/floor-450_synthetic-0.3_pretrain-4 | +0.177 [-0.093, +0.568] | +0.166 [-0.503, +0.846] | -0.480 [-0.825, -0.157] |
| synthesis | cod-p10/floor-450_synthetic-0.6_pretrain-48 | cod-p10/floor-450_synthetic-0.3_pretrain-48 | +1.564 [+1.151, +2.041] | +6.160 [-1.064, +20.284] | +2.182 [+1.840, +2.638] |
| metadata | metadata-p10/floor-450_synthetic-0.3_pretrain-48 | cod-p10/floor-450_synthetic-0.3_pretrain-48 | +0.492 [+0.102, +1.048] | +5.779 [-2.231, +20.087] | +1.590 [+1.139, +2.069] |
| patience | cod-p20/floor-450_synthetic-0.3_pretrain-48 | cod-p10/floor-450_synthetic-0.3_pretrain-48 | +0.529 [+0.173, +0.985] | +6.602 [+0.082, +20.724] | -0.011 [-0.321, +0.368] |
| pretraining | cod-p10/floor-450_synthetic-0.6_pretrain-48 | cod-p10/floor-450_synthetic-0.6_pretrain-4 | +0.259 [-0.273, +0.607] | -0.099 [-3.201, +2.847] | +0.366 [+0.012, +0.791] |
| metadata | metadata-p10/floor-450_synthetic-0.6_pretrain-4 | cod-p10/floor-450_synthetic-0.6_pretrain-4 | -0.599 [-1.086, -0.181] | -0.210 [-2.509, +1.074] | +0.151 [-0.235, +0.589] |
| patience | cod-p20/floor-450_synthetic-0.6_pretrain-4 | cod-p10/floor-450_synthetic-0.6_pretrain-4 | +0.273 [-0.000, +0.605] | +0.002 [-0.165, +0.120] | -0.057 [-0.295, +0.286] |
| metadata | metadata-p10/floor-450_synthetic-0.6_pretrain-48 | cod-p10/floor-450_synthetic-0.6_pretrain-48 | -0.913 [-1.331, -0.404] | +0.322 [-2.550, +3.262] | -0.086 [-0.576, +0.269] |
| patience | cod-p20/floor-450_synthetic-0.6_pretrain-48 | cod-p10/floor-450_synthetic-0.6_pretrain-48 | +0.243 [-0.081, +0.595] | +0.594 [-0.556, +2.896] | -0.223 [-0.556, +0.032] |
| pretraining | metadata-p10/floor-0_synthetic-0.3_pretrain-48 | metadata-p10/floor-0_synthetic-0.3_pretrain-4 | +1.100 [+0.704, +1.604] | -0.320 [-1.005, +0.205] | +1.235 [+0.736, +1.603] |
| synthesis | metadata-p10/floor-0_synthetic-0.6_pretrain-4 | metadata-p10/floor-0_synthetic-0.3_pretrain-4 | -0.441 [-0.799, +0.072] | -0.316 [-0.936, +0.197] | +0.197 [-0.157, +0.673] |
| floor | metadata-p10/floor-0_synthetic-0.3_pretrain-4 | metadata-p10/floor-450_synthetic-0.3_pretrain-4 | +1.289 [+0.795, +1.733] | +1.117 [-0.026, +3.503] | +0.595 [+0.187, +1.034] |
| patience | metadata-p20/floor-0_synthetic-0.3_pretrain-4 | metadata-p10/floor-0_synthetic-0.3_pretrain-4 | +1.467 [+1.127, +1.851] | -0.091 [-0.713, +0.440] | +0.609 [+0.280, +0.975] |
| synthesis | metadata-p10/floor-0_synthetic-0.6_pretrain-48 | metadata-p10/floor-0_synthetic-0.3_pretrain-48 | -0.317 [-0.773, +0.144] | -2.432 [-7.644, +0.268] | +0.762 [+0.423, +1.171] |
| floor | metadata-p10/floor-0_synthetic-0.3_pretrain-48 | metadata-p10/floor-450_synthetic-0.3_pretrain-48 | +2.005 [+1.559, +2.438] | +0.968 [-0.157, +3.253] | +0.542 [+0.036, +0.924] |
| patience | metadata-p20/floor-0_synthetic-0.3_pretrain-48 | metadata-p10/floor-0_synthetic-0.3_pretrain-48 | +0.245 [-0.108, +0.569] | +1.116 [+0.171, +3.516] | +0.248 [-0.024, +0.554] |
| pretraining | metadata-p10/floor-0_synthetic-0.6_pretrain-48 | metadata-p10/floor-0_synthetic-0.6_pretrain-4 | +1.225 [+0.748, +1.642] | -2.435 [-7.612, +0.220] | +1.800 [+1.294, +2.105] |
| floor | metadata-p10/floor-0_synthetic-0.6_pretrain-4 | metadata-p10/floor-450_synthetic-0.6_pretrain-4 | +0.248 [-0.187, +0.707] | +0.702 [-0.544, +3.062] | -0.872 [-1.218, -0.456] |
| patience | metadata-p20/floor-0_synthetic-0.6_pretrain-4 | metadata-p10/floor-0_synthetic-0.6_pretrain-4 | +1.010 [+0.658, +1.312] | +0.283 [-0.026, +0.615] | +0.723 [+0.400, +1.053] |
| floor | metadata-p10/floor-0_synthetic-0.6_pretrain-48 | metadata-p10/floor-450_synthetic-0.6_pretrain-48 | +1.528 [+1.081, +1.974] | -2.167 [-7.378, +0.654] | +0.798 [+0.409, +1.166] |
| patience | metadata-p20/floor-0_synthetic-0.6_pretrain-48 | metadata-p10/floor-0_synthetic-0.6_pretrain-48 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| pretraining | metadata-p10/floor-450_synthetic-0.3_pretrain-48 | metadata-p10/floor-450_synthetic-0.3_pretrain-4 | +0.385 [-0.029, +0.877] | -0.171 [-0.780, +0.450] | +1.288 [+0.820, +1.775] |
| synthesis | metadata-p10/floor-450_synthetic-0.6_pretrain-4 | metadata-p10/floor-450_synthetic-0.3_pretrain-4 | +0.600 [+0.201, +1.078] | +0.099 [-0.631, +0.835] | +1.665 [+1.296, +2.110] |
| patience | metadata-p20/floor-450_synthetic-0.3_pretrain-4 | metadata-p10/floor-450_synthetic-0.3_pretrain-4 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| synthesis | metadata-p10/floor-450_synthetic-0.6_pretrain-48 | metadata-p10/floor-450_synthetic-0.3_pretrain-48 | +0.160 [-0.283, +0.593] | +0.704 [-0.419, +2.847] | +0.506 [+0.077, +0.896] |
| patience | metadata-p20/floor-450_synthetic-0.3_pretrain-48 | metadata-p10/floor-450_synthetic-0.3_pretrain-48 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| pretraining | metadata-p10/floor-450_synthetic-0.6_pretrain-48 | metadata-p10/floor-450_synthetic-0.6_pretrain-4 | -0.055 [-0.511, +0.364] | +0.434 [-0.684, +2.555] | +0.130 [-0.351, +0.494] |
| patience | metadata-p20/floor-450_synthetic-0.6_pretrain-4 | metadata-p10/floor-450_synthetic-0.6_pretrain-4 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| patience | metadata-p20/floor-450_synthetic-0.6_pretrain-48 | metadata-p10/floor-450_synthetic-0.6_pretrain-48 | +0.767 [+0.351, +1.097] | -0.529 [-1.699, +0.103] | +0.414 [+0.067, +0.742] |
| pretraining | cod-p20/floor-0_synthetic-0.3_pretrain-48 | cod-p20/floor-0_synthetic-0.3_pretrain-4 | +0.379 [-0.078, +0.835] | -1.788 [-4.248, -0.400] | -0.160 [-0.658, +0.260] |
| synthesis | cod-p20/floor-0_synthetic-0.6_pretrain-4 | cod-p20/floor-0_synthetic-0.3_pretrain-4 | +0.143 [-0.270, +0.580] | -1.453 [-4.026, -0.192] | -0.522 [-0.929, -0.067] |
| floor | cod-p20/floor-0_synthetic-0.3_pretrain-4 | cod-p20/floor-450_synthetic-0.3_pretrain-4 | +1.368 [+0.925, +1.838] | +0.750 [+0.003, +2.043] | +0.737 [+0.277, +1.293] |
| metadata | metadata-p20/floor-0_synthetic-0.3_pretrain-4 | cod-p20/floor-0_synthetic-0.3_pretrain-4 | +0.459 [-0.016, +0.914] | -1.494 [-4.043, -0.192] | -0.047 [-0.410, +0.394] |
| synthesis | cod-p20/floor-0_synthetic-0.6_pretrain-48 | cod-p20/floor-0_synthetic-0.3_pretrain-48 | -0.564 [-0.964, -0.074] | +0.359 [-0.620, +1.845] | +1.061 [+0.699, +1.544] |
| floor | cod-p20/floor-0_synthetic-0.3_pretrain-48 | cod-p20/floor-450_synthetic-0.3_pretrain-48 | +2.255 [+1.800, +2.693] | +0.080 [-0.611, +0.591] | +1.405 [+0.929, +1.736] |
| metadata | metadata-p20/floor-0_synthetic-0.3_pretrain-48 | cod-p20/floor-0_synthetic-0.3_pretrain-48 | -0.043 [-0.471, +0.424] | +1.181 [-0.068, +3.440] | +0.987 [+0.542, +1.440] |
| pretraining | cod-p20/floor-0_synthetic-0.6_pretrain-48 | cod-p20/floor-0_synthetic-0.6_pretrain-4 | -0.328 [-0.763, +0.168] | +0.024 [-2.205, +2.687] | +1.424 [+0.972, +1.877] |
| floor | cod-p20/floor-0_synthetic-0.6_pretrain-4 | cod-p20/floor-450_synthetic-0.6_pretrain-4 | +0.969 [+0.452, +1.309] | +0.756 [+0.279, +1.526] | -0.728 [-1.152, -0.389] |
| metadata | metadata-p20/floor-0_synthetic-0.6_pretrain-4 | cod-p20/floor-0_synthetic-0.6_pretrain-4 | -0.583 [-1.018, -0.114] | +0.016 [-0.422, +0.534] | +0.787 [+0.405, +1.290] |
| floor | cod-p20/floor-0_synthetic-0.6_pretrain-48 | cod-p20/floor-450_synthetic-0.6_pretrain-48 | +0.413 [-0.009, +0.919] | +0.287 [-0.659, +1.245] | +0.495 [+0.134, +0.879] |
| metadata | metadata-p20/floor-0_synthetic-0.6_pretrain-48 | cod-p20/floor-0_synthetic-0.6_pretrain-48 | -0.040 [-0.538, +0.433] | -2.726 [-8.124, +0.476] | +0.440 [-0.025, +0.803] |
| pretraining | cod-p20/floor-450_synthetic-0.3_pretrain-48 | cod-p20/floor-450_synthetic-0.3_pretrain-4 | -0.509 [-0.935, -0.046] | -1.118 [-3.342, +0.149] | -0.827 [-1.202, -0.279] |
| synthesis | cod-p20/floor-450_synthetic-0.6_pretrain-4 | cod-p20/floor-450_synthetic-0.3_pretrain-4 | +0.541 [+0.228, +1.077] | -1.459 [-3.879, -0.286] | +0.942 [+0.655, +1.460] |
| metadata | metadata-p20/floor-450_synthetic-0.3_pretrain-4 | cod-p20/floor-450_synthetic-0.3_pretrain-4 | -0.930 [-1.402, -0.440] | -1.771 [-4.836, -0.058] | -0.514 [-0.960, +0.045] |
| synthesis | cod-p20/floor-450_synthetic-0.6_pretrain-48 | cod-p20/floor-450_synthetic-0.3_pretrain-48 | +1.279 [+0.839, +1.704] | +0.153 [-0.921, +1.711] | +1.971 [+1.527, +2.369] |
| metadata | metadata-p20/floor-450_synthetic-0.3_pretrain-48 | cod-p20/floor-450_synthetic-0.3_pretrain-48 | -0.037 [-0.477, +0.460] | -0.823 [-4.145, +2.007] | +1.602 [+1.139, +2.030] |
| pretraining | cod-p20/floor-450_synthetic-0.6_pretrain-48 | cod-p20/floor-450_synthetic-0.6_pretrain-4 | +0.229 [-0.332, +0.585] | +0.494 [-1.719, +3.082] | +0.201 [-0.245, +0.554] |
| metadata | metadata-p20/floor-450_synthetic-0.6_pretrain-4 | cod-p20/floor-450_synthetic-0.6_pretrain-4 | -0.872 [-1.368, -0.477] | -0.212 [-2.529, +1.075] | +0.209 [-0.240, +0.604] |
| metadata | metadata-p20/floor-450_synthetic-0.6_pretrain-48 | cod-p20/floor-450_synthetic-0.6_pretrain-48 | -0.389 [-0.861, +0.052] | -0.801 [-3.248, +0.619] | +0.551 [+0.087, +0.941] |
| pretraining | metadata-p20/floor-0_synthetic-0.3_pretrain-48 | metadata-p20/floor-0_synthetic-0.3_pretrain-4 | -0.122 [-0.558, +0.362] | +0.887 [-0.204, +3.259] | +0.874 [+0.374, +1.235] |
| synthesis | metadata-p20/floor-0_synthetic-0.6_pretrain-4 | metadata-p20/floor-0_synthetic-0.3_pretrain-4 | -0.899 [-1.291, -0.412] | +0.057 [-0.238, +0.372] | +0.311 [-0.064, +0.781] |
| floor | metadata-p20/floor-0_synthetic-0.3_pretrain-4 | metadata-p20/floor-450_synthetic-0.3_pretrain-4 | +2.757 [+2.297, +3.202] | +1.026 [-0.005, +3.277] | +1.204 [+0.782, +1.707] |
| synthesis | metadata-p20/floor-0_synthetic-0.6_pretrain-48 | metadata-p20/floor-0_synthetic-0.3_pretrain-48 | -0.561 [-1.001, -0.092] | -3.547 [-8.825, -0.399] | +0.514 [+0.137, +0.928] |
| floor | metadata-p20/floor-0_synthetic-0.3_pretrain-48 | metadata-p20/floor-450_synthetic-0.3_pretrain-48 | +2.250 [+1.756, +2.694] | +2.084 [+0.440, +5.121] | +0.790 [+0.281, +1.216] |
| pretraining | metadata-p20/floor-0_synthetic-0.6_pretrain-48 | metadata-p20/floor-0_synthetic-0.6_pretrain-4 | +0.215 [-0.255, +0.670] | -2.718 [-7.875, -0.057] | +1.077 [+0.557, +1.377] |
| floor | metadata-p20/floor-0_synthetic-0.6_pretrain-4 | metadata-p20/floor-450_synthetic-0.6_pretrain-4 | +1.258 [+0.790, +1.694] | +0.984 [-0.245, +3.346] | -0.149 [-0.488, +0.266] |
| floor | metadata-p20/floor-0_synthetic-0.6_pretrain-48 | metadata-p20/floor-450_synthetic-0.6_pretrain-48 | +0.761 [+0.357, +1.251] | -1.638 [-6.830, +1.062] | +0.384 [+0.018, +0.755] |
| pretraining | metadata-p20/floor-450_synthetic-0.3_pretrain-48 | metadata-p20/floor-450_synthetic-0.3_pretrain-4 | +0.385 [-0.029, +0.877] | -0.171 [-0.780, +0.450] | +1.288 [+0.820, +1.775] |
| synthesis | metadata-p20/floor-450_synthetic-0.6_pretrain-4 | metadata-p20/floor-450_synthetic-0.3_pretrain-4 | +0.600 [+0.201, +1.078] | +0.099 [-0.631, +0.835] | +1.665 [+1.296, +2.110] |
| synthesis | metadata-p20/floor-450_synthetic-0.6_pretrain-48 | metadata-p20/floor-450_synthetic-0.3_pretrain-48 | +0.927 [+0.421, +1.322] | +0.175 [-1.330, +2.040] | +0.921 [+0.471, +1.301] |
| pretraining | metadata-p20/floor-450_synthetic-0.6_pretrain-48 | metadata-p20/floor-450_synthetic-0.6_pretrain-4 | +0.712 [+0.202, +1.086] | -0.095 [-1.580, +1.795] | +0.544 [+0.084, +0.867] |

All within-campaign recipe pairs, all metrics and sensitivity intervals are in [paired_intervals.csv](paired_intervals.csv).
