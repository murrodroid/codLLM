# Paired publication intervals

10,000 paired connected-COD/source-record multinomial bootstrap replicates; 95% percentile intervals. Differences are first minus second, in percentage points.

Conditional on the selected models and observed archive panel. Validation selection and training-seed variation are not covered; intervals are exploratory and not multiplicity-adjusted.

The published metrics are recomputed in every draw (pooled macro: reference ∪ predictions per model; source reference macro: reference labels per archive). Positive Bayesian weights retain observed support. Ordinary multinomial draws may omit rare labels; fixed-support sensitivity metrics assign absent labels zero F1. The CSV/JSON retain both definitions and rare-label diagnostics.

Ordinary n-out-of-n paired connected COD/source-record cluster bootstrap; percentile confidence intervals.

Bayesian draws assign normalized independent Exp(1) weights to connected groups, equivalent to Dirichlet(1,...,1) weights. Every model receives exactly the same group weights in each draw. We apply [Rubin's Bayesian bootstrap](https://people.eecs.berkeley.edu/~jordan/sail/readings/rubin.pdf) to the connected COD/source-record units, retaining row-frequency and equal-source metric definitions.

The central interval of bootstrap metric draws need not contain the original point estimate: F1 is nonlinear and sparse support can shift its resampling distribution. Bayesian weighting does not supply missing evidence within labels represented by a single group. Treat these intervals as conditional exploratory evidence, not calibrated uncertainty over unseen label cases or new archives.

## Matched comparisons

| Change | First | Second | Pooled macro F1 Δ [95% interval] | Micro F1 Δ [95% interval] | Equal-source ref. macro F1 Δ [95% interval] |
|---|---|---|---:|---:|---:|
| pretraining | cod-p10/floor-0_synthetic-0.3_pretrain-48 | cod-p10/floor-0_synthetic-0.3_pretrain-4 | -0.011 [-1.039, +1.041] | -1.403 [-3.427, +0.046] | -0.154 [-1.152, +0.838] |
| synthesis | cod-p10/floor-0_synthetic-0.6_pretrain-4 | cod-p10/floor-0_synthetic-0.3_pretrain-4 | -0.849 [-1.658, +0.235] | -1.791 [-4.189, -0.296] | -0.684 [-1.572, +0.411] |
| floor | cod-p10/floor-0_synthetic-0.3_pretrain-4 | cod-p10/floor-450_synthetic-0.3_pretrain-4 | +1.544 [+0.447, +2.575] | +0.917 [-0.126, +2.260] | +0.257 [-0.593, +1.673] |
| metadata | metadata-p10/floor-0_synthetic-0.3_pretrain-4 | cod-p10/floor-0_synthetic-0.3_pretrain-4 | -1.008 [-2.062, +0.015] | -1.403 [-3.847, +0.275] | -0.656 [-1.645, +0.433] |
| patience | cod-p20/floor-0_synthetic-0.3_pretrain-4 | cod-p10/floor-0_synthetic-0.3_pretrain-4 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| synthesis | cod-p10/floor-0_synthetic-0.6_pretrain-48 | cod-p10/floor-0_synthetic-0.3_pretrain-48 | -0.174 [-1.252, +0.712] | -0.026 [-0.495, +0.520] | +1.055 [+0.006, +1.903] |
| floor | cod-p10/floor-0_synthetic-0.3_pretrain-48 | cod-p10/floor-450_synthetic-0.3_pretrain-48 | +2.394 [+1.477, +3.608] | +7.067 [-0.428, +20.263] | +1.400 [+0.031, +2.341] |
| metadata | metadata-p10/floor-0_synthetic-0.3_pretrain-48 | cod-p10/floor-0_synthetic-0.3_pretrain-48 | +0.103 [-1.113, +1.019] | -0.320 [-2.783, +1.915] | +0.732 [-0.602, +1.513] |
| patience | cod-p20/floor-0_synthetic-0.3_pretrain-48 | cod-p10/floor-0_synthetic-0.3_pretrain-48 | +0.390 [-0.304, +1.118] | -0.385 [-1.729, +0.579] | -0.006 [-0.719, +0.931] |
| pretraining | cod-p10/floor-0_synthetic-0.6_pretrain-48 | cod-p10/floor-0_synthetic-0.6_pretrain-4 | +0.664 [-0.614, +1.463] | +0.363 [-1.873, +2.842] | +1.586 [+0.312, +2.488] |
| floor | cod-p10/floor-0_synthetic-0.6_pretrain-4 | cod-p10/floor-450_synthetic-0.6_pretrain-4 | +0.250 [-0.921, +1.166] | +0.420 [-0.115, +1.120] | -0.947 [-1.644, +0.399] |
| metadata | metadata-p10/floor-0_synthetic-0.6_pretrain-4 | cod-p10/floor-0_synthetic-0.6_pretrain-4 | -0.601 [-1.559, +0.379] | +0.072 [-0.300, +0.476] | +0.226 [-0.731, +1.208] |
| patience | cod-p20/floor-0_synthetic-0.6_pretrain-4 | cod-p10/floor-0_synthetic-0.6_pretrain-4 | +0.992 [+0.244, +1.777] | +0.338 [+0.136, +0.596] | +0.162 [-0.439, +0.930] |
| floor | cod-p10/floor-0_synthetic-0.6_pretrain-48 | cod-p10/floor-450_synthetic-0.6_pretrain-48 | +0.655 [-0.440, +1.546] | +0.881 [-0.622, +3.112] | +0.272 [-0.543, +1.201] |
| metadata | metadata-p10/floor-0_synthetic-0.6_pretrain-48 | cod-p10/floor-0_synthetic-0.6_pretrain-48 | -0.040 [-0.772, +1.380] | -2.726 [-7.838, +1.007] | +0.440 [-0.724, +1.364] |
| patience | cod-p20/floor-0_synthetic-0.6_pretrain-48 | cod-p10/floor-0_synthetic-0.6_pretrain-48 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| pretraining | cod-p10/floor-450_synthetic-0.3_pretrain-48 | cod-p10/floor-450_synthetic-0.3_pretrain-4 | -0.860 [-2.074, +0.019] | -7.553 [-20.594, +0.310] | -1.296 [-1.990, +0.393] |
| synthesis | cod-p10/floor-450_synthetic-0.6_pretrain-4 | cod-p10/floor-450_synthetic-0.3_pretrain-4 | +0.446 [-0.266, +1.650] | -1.294 [-3.565, +0.176] | +0.519 [-0.537, +1.654] |
| metadata | metadata-p10/floor-450_synthetic-0.3_pretrain-4 | cod-p10/floor-450_synthetic-0.3_pretrain-4 | -0.753 [-1.767, +0.375] | -1.604 [-4.442, +0.448] | -0.994 [-1.627, +0.572] |
| patience | cod-p20/floor-450_synthetic-0.3_pretrain-4 | cod-p10/floor-450_synthetic-0.3_pretrain-4 | +0.177 [-0.655, +0.828] | +0.166 [-0.491, +0.830] | -0.480 [-1.096, +0.328] |
| synthesis | cod-p10/floor-450_synthetic-0.6_pretrain-48 | cod-p10/floor-450_synthetic-0.3_pretrain-48 | +1.564 [+0.718, +2.715] | +6.160 [-2.205, +19.527] | +2.182 [+0.675, +2.961] |
| metadata | metadata-p10/floor-450_synthetic-0.3_pretrain-48 | cod-p10/floor-450_synthetic-0.3_pretrain-48 | +0.492 [-0.444, +1.727] | +5.779 [-3.348, +19.399] | +1.590 [+0.275, +2.648] |
| patience | cod-p20/floor-450_synthetic-0.3_pretrain-48 | cod-p10/floor-450_synthetic-0.3_pretrain-48 | +0.529 [-0.292, +1.485] | +6.602 [-0.400, +19.798] | -0.011 [-0.949, +0.776] |
| pretraining | cod-p10/floor-450_synthetic-0.6_pretrain-48 | cod-p10/floor-450_synthetic-0.6_pretrain-4 | +0.259 [-0.986, +0.999] | -0.099 [-3.019, +2.783] | +0.366 [-0.557, +1.463] |
| metadata | metadata-p10/floor-450_synthetic-0.6_pretrain-4 | cod-p10/floor-450_synthetic-0.6_pretrain-4 | -0.599 [-1.728, +0.266] | -0.210 [-2.350, +1.195] | +0.151 [-0.474, +1.656] |
| patience | cod-p20/floor-450_synthetic-0.6_pretrain-4 | cod-p10/floor-450_synthetic-0.6_pretrain-4 | +0.273 [-0.519, +0.910] | +0.002 [-0.154, +0.129] | -0.057 [-0.610, +0.915] |
| metadata | metadata-p10/floor-450_synthetic-0.6_pretrain-48 | cod-p10/floor-450_synthetic-0.6_pretrain-48 | -0.913 [-1.910, +0.163] | +0.322 [-2.452, +3.173] | -0.086 [-1.101, +0.971] |
| patience | cod-p20/floor-450_synthetic-0.6_pretrain-48 | cod-p10/floor-450_synthetic-0.6_pretrain-48 | +0.243 [-0.632, +0.874] | +0.594 [-0.719, +2.772] | -0.223 [-0.936, +0.381] |
| pretraining | metadata-p10/floor-0_synthetic-0.3_pretrain-48 | metadata-p10/floor-0_synthetic-0.3_pretrain-4 | +1.100 [+0.022, +1.998] | -0.320 [-0.956, +0.215] | +1.235 [-0.116, +1.967] |
| synthesis | metadata-p10/floor-0_synthetic-0.6_pretrain-4 | metadata-p10/floor-0_synthetic-0.3_pretrain-4 | -0.441 [-1.199, +0.675] | -0.316 [-0.911, +0.193] | +0.197 [-0.761, +1.215] |
| floor | metadata-p10/floor-0_synthetic-0.3_pretrain-4 | metadata-p10/floor-450_synthetic-0.3_pretrain-4 | +1.289 [+0.162, +2.232] | +1.117 [-0.195, +3.314] | +0.595 [-0.527, +1.415] |
| patience | metadata-p20/floor-0_synthetic-0.3_pretrain-4 | metadata-p10/floor-0_synthetic-0.3_pretrain-4 | +1.467 [+0.617, +2.243] | -0.091 [-0.713, +0.444] | +0.609 [+0.083, +1.590] |
| synthesis | metadata-p10/floor-0_synthetic-0.6_pretrain-48 | metadata-p10/floor-0_synthetic-0.3_pretrain-48 | -0.317 [-0.985, +1.139] | -2.432 [-7.241, +0.686] | +0.762 [-0.101, +1.761] |
| floor | metadata-p10/floor-0_synthetic-0.3_pretrain-48 | metadata-p10/floor-450_synthetic-0.3_pretrain-48 | +2.005 [+0.857, +2.896] | +0.968 [-0.290, +3.067] | +0.542 [-0.773, +1.103] |
| patience | metadata-p20/floor-0_synthetic-0.3_pretrain-48 | metadata-p10/floor-0_synthetic-0.3_pretrain-48 | +0.245 [-0.368, +1.098] | +1.116 [+0.057, +3.271] | +0.248 [-0.190, +1.223] |
| pretraining | metadata-p10/floor-0_synthetic-0.6_pretrain-48 | metadata-p10/floor-0_synthetic-0.6_pretrain-4 | +1.225 [+0.323, +2.375] | -2.435 [-7.220, +0.633] | +1.800 [+0.536, +2.432] |
| floor | metadata-p10/floor-0_synthetic-0.6_pretrain-4 | metadata-p10/floor-450_synthetic-0.6_pretrain-4 | +0.248 [-0.732, +1.266] | +0.702 [-0.701, +2.845] | -0.872 [-1.875, -0.110] |
| patience | metadata-p20/floor-0_synthetic-0.6_pretrain-4 | metadata-p10/floor-0_synthetic-0.6_pretrain-4 | +1.010 [+0.277, +1.794] | +0.283 [-0.026, +0.605] | +0.723 [+0.158, +1.494] |
| floor | metadata-p10/floor-0_synthetic-0.6_pretrain-48 | metadata-p10/floor-450_synthetic-0.6_pretrain-48 | +1.528 [+0.701, +2.792] | -2.167 [-7.041, +1.058] | +0.798 [-0.181, +1.602] |
| patience | metadata-p20/floor-0_synthetic-0.6_pretrain-48 | metadata-p10/floor-0_synthetic-0.6_pretrain-48 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| pretraining | metadata-p10/floor-450_synthetic-0.3_pretrain-48 | metadata-p10/floor-450_synthetic-0.3_pretrain-4 | +0.385 [-0.748, +1.342] | -0.171 [-0.777, +0.453] | +1.288 [+0.117, +2.235] |
| synthesis | metadata-p10/floor-450_synthetic-0.6_pretrain-4 | metadata-p10/floor-450_synthetic-0.3_pretrain-4 | +0.600 [-0.344, +1.617] | +0.099 [-0.626, +0.822] | +1.665 [+0.765, +2.580] |
| patience | metadata-p20/floor-450_synthetic-0.3_pretrain-4 | metadata-p10/floor-450_synthetic-0.3_pretrain-4 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| synthesis | metadata-p10/floor-450_synthetic-0.6_pretrain-48 | metadata-p10/floor-450_synthetic-0.3_pretrain-48 | +0.160 [-0.809, +1.218] | +0.704 [-0.544, +2.681] | +0.506 [-0.565, +1.132] |
| patience | metadata-p20/floor-450_synthetic-0.3_pretrain-48 | metadata-p10/floor-450_synthetic-0.3_pretrain-48 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| pretraining | metadata-p10/floor-450_synthetic-0.6_pretrain-48 | metadata-p10/floor-450_synthetic-0.6_pretrain-4 | -0.055 [-1.150, +0.850] | +0.434 [-0.800, +2.411] | +0.130 [-1.148, +0.693] |
| patience | metadata-p20/floor-450_synthetic-0.6_pretrain-4 | metadata-p10/floor-450_synthetic-0.6_pretrain-4 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| patience | metadata-p20/floor-450_synthetic-0.6_pretrain-48 | metadata-p10/floor-450_synthetic-0.6_pretrain-48 | +0.767 [-0.148, +1.587] | -0.529 [-1.621, +0.186] | +0.414 [-0.266, +1.125] |
| pretraining | cod-p20/floor-0_synthetic-0.3_pretrain-48 | cod-p20/floor-0_synthetic-0.3_pretrain-4 | +0.379 [-0.597, +1.449] | -1.788 [-4.051, -0.123] | -0.160 [-1.133, +1.028] |
| synthesis | cod-p20/floor-0_synthetic-0.6_pretrain-4 | cod-p20/floor-0_synthetic-0.3_pretrain-4 | +0.143 [-0.685, +1.295] | -1.453 [-3.809, +0.047] | -0.522 [-1.324, +0.666] |
| floor | cod-p20/floor-0_synthetic-0.3_pretrain-4 | cod-p20/floor-450_synthetic-0.3_pretrain-4 | +1.368 [+0.378, +2.506] | +0.750 [-0.063, +1.910] | +0.737 [-0.167, +2.039] |
| metadata | metadata-p20/floor-0_synthetic-0.3_pretrain-4 | cod-p20/floor-0_synthetic-0.3_pretrain-4 | +0.459 [-0.594, +1.416] | -1.494 [-3.882, +0.054] | -0.047 [-0.730, +1.195] |
| synthesis | cod-p20/floor-0_synthetic-0.6_pretrain-48 | cod-p20/floor-0_synthetic-0.3_pretrain-48 | -0.564 [-1.658, +0.271] | +0.359 [-0.661, +1.755] | +1.061 [-0.198, +1.929] |
| floor | cod-p20/floor-0_synthetic-0.3_pretrain-48 | cod-p20/floor-450_synthetic-0.3_pretrain-48 | +2.255 [+1.361, +3.380] | +0.080 [-0.564, +0.645] | +1.405 [+0.333, +2.391] |
| metadata | metadata-p20/floor-0_synthetic-0.3_pretrain-48 | cod-p20/floor-0_synthetic-0.3_pretrain-48 | -0.043 [-1.112, +0.941] | +1.181 [-0.275, +3.283] | +0.987 [-0.224, +1.993] |
| pretraining | cod-p20/floor-0_synthetic-0.6_pretrain-48 | cod-p20/floor-0_synthetic-0.6_pretrain-4 | -0.328 [-1.655, +0.505] | +0.024 [-2.220, +2.509] | +1.424 [+0.020, +2.282] |
| floor | cod-p20/floor-0_synthetic-0.6_pretrain-4 | cod-p20/floor-450_synthetic-0.6_pretrain-4 | +0.969 [-0.122, +1.984] | +0.756 [+0.224, +1.485] | -0.728 [-1.442, +0.401] |
| metadata | metadata-p20/floor-0_synthetic-0.6_pretrain-4 | cod-p20/floor-0_synthetic-0.6_pretrain-4 | -0.583 [-1.579, +0.431] | +0.016 [-0.428, +0.505] | +0.787 [-0.221, +1.871] |
| floor | cod-p20/floor-0_synthetic-0.6_pretrain-48 | cod-p20/floor-450_synthetic-0.6_pretrain-48 | +0.413 [-0.628, +1.524] | +0.287 [-0.639, +1.229] | +0.495 [-0.367, +1.548] |
| metadata | metadata-p20/floor-0_synthetic-0.6_pretrain-48 | cod-p20/floor-0_synthetic-0.6_pretrain-48 | -0.040 [-0.772, +1.380] | -2.726 [-7.838, +1.007] | +0.440 [-0.724, +1.364] |
| pretraining | cod-p20/floor-450_synthetic-0.3_pretrain-48 | cod-p20/floor-450_synthetic-0.3_pretrain-4 | -0.509 [-1.504, +0.487] | -1.118 [-3.227, +0.373] | -0.827 [-1.608, +0.617] |
| synthesis | cod-p20/floor-450_synthetic-0.6_pretrain-4 | cod-p20/floor-450_synthetic-0.3_pretrain-4 | +0.541 [-0.224, +1.813] | -1.459 [-3.698, -0.087] | +0.942 [+0.059, +2.167] |
| metadata | metadata-p20/floor-450_synthetic-0.3_pretrain-4 | cod-p20/floor-450_synthetic-0.3_pretrain-4 | -0.930 [-1.836, +0.264] | -1.771 [-4.633, +0.269] | -0.514 [-1.298, +1.053] |
| synthesis | cod-p20/floor-450_synthetic-0.6_pretrain-48 | cod-p20/floor-450_synthetic-0.3_pretrain-48 | +1.279 [+0.206, +2.251] | +0.153 [-0.990, +1.602] | +1.971 [+0.537, +2.725] |
| metadata | metadata-p20/floor-450_synthetic-0.3_pretrain-48 | cod-p20/floor-450_synthetic-0.3_pretrain-48 | -0.037 [-1.013, +1.095] | -0.823 [-4.035, +2.036] | +1.602 [+0.418, +2.678] |
| pretraining | cod-p20/floor-450_synthetic-0.6_pretrain-48 | cod-p20/floor-450_synthetic-0.6_pretrain-4 | +0.229 [-1.104, +0.945] | +0.494 [-1.741, +2.926] | +0.201 [-1.054, +1.130] |
| metadata | metadata-p20/floor-450_synthetic-0.6_pretrain-4 | cod-p20/floor-450_synthetic-0.6_pretrain-4 | -0.872 [-1.944, +0.066] | -0.212 [-2.364, +1.189] | +0.209 [-0.577, +1.505] |
| metadata | metadata-p20/floor-450_synthetic-0.6_pretrain-48 | cod-p20/floor-450_synthetic-0.6_pretrain-48 | -0.389 [-1.294, +0.682] | -0.801 [-3.051, +0.755] | +0.551 [-0.480, +1.731] |
| pretraining | metadata-p20/floor-0_synthetic-0.3_pretrain-48 | metadata-p20/floor-0_synthetic-0.3_pretrain-4 | -0.122 [-1.063, +0.959] | +0.887 [-0.365, +3.057] | +0.874 [-0.384, +1.582] |
| synthesis | metadata-p20/floor-0_synthetic-0.6_pretrain-4 | metadata-p20/floor-0_synthetic-0.3_pretrain-4 | -0.899 [-1.619, +0.328] | +0.057 [-0.246, +0.364] | +0.311 [-0.643, +1.113] |
| floor | metadata-p20/floor-0_synthetic-0.3_pretrain-4 | metadata-p20/floor-450_synthetic-0.3_pretrain-4 | +2.757 [+1.571, +3.665] | +1.026 [-0.178, +3.172] | +1.204 [+0.269, +2.260] |
| synthesis | metadata-p20/floor-0_synthetic-0.6_pretrain-48 | metadata-p20/floor-0_synthetic-0.3_pretrain-48 | -0.561 [-1.338, +0.725] | -3.547 [-8.534, +0.226] | +0.514 [-0.737, +1.354] |
| floor | metadata-p20/floor-0_synthetic-0.3_pretrain-48 | metadata-p20/floor-450_synthetic-0.3_pretrain-48 | +2.250 [+1.194, +3.292] | +2.084 [+0.103, +4.946] | +0.790 [-0.323, +1.711] |
| pretraining | metadata-p20/floor-0_synthetic-0.6_pretrain-48 | metadata-p20/floor-0_synthetic-0.6_pretrain-4 | +0.215 [-0.765, +1.327] | -2.718 [-7.471, +0.345] | +1.077 [-0.268, +1.617] |
| floor | metadata-p20/floor-0_synthetic-0.6_pretrain-4 | metadata-p20/floor-450_synthetic-0.6_pretrain-4 | +1.258 [+0.293, +2.327] | +0.984 [-0.416, +3.143] | -0.149 [-1.026, +0.673] |
| floor | metadata-p20/floor-0_synthetic-0.6_pretrain-48 | metadata-p20/floor-450_synthetic-0.6_pretrain-48 | +0.761 [-0.024, +2.083] | -1.638 [-6.522, +1.395] | +0.384 [-0.550, +1.136] |
| pretraining | metadata-p20/floor-450_synthetic-0.3_pretrain-48 | metadata-p20/floor-450_synthetic-0.3_pretrain-4 | +0.385 [-0.748, +1.342] | -0.171 [-0.777, +0.453] | +1.288 [+0.117, +2.235] |
| synthesis | metadata-p20/floor-450_synthetic-0.6_pretrain-4 | metadata-p20/floor-450_synthetic-0.3_pretrain-4 | +0.600 [-0.344, +1.617] | +0.099 [-0.626, +0.822] | +1.665 [+0.765, +2.580] |
| synthesis | metadata-p20/floor-450_synthetic-0.6_pretrain-48 | metadata-p20/floor-450_synthetic-0.3_pretrain-48 | +0.927 [-0.087, +1.900] | +0.175 [-1.335, +1.949] | +0.921 [-0.225, +1.603] |
| pretraining | metadata-p20/floor-450_synthetic-0.6_pretrain-48 | metadata-p20/floor-450_synthetic-0.6_pretrain-4 | +0.712 [-0.413, +1.541] | -0.095 [-1.608, +1.626] | +0.544 [-0.722, +1.128] |

All within-campaign recipe pairs, all metrics and sensitivity intervals are in [paired_intervals.csv](paired_intervals_multinomial.csv).
