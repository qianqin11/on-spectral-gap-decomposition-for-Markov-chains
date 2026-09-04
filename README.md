# Simulation supplement for Section 7.2

This package contains the code, numerical results, and six figure panels for
the simulated experiment in Section 7.2 of *On spectral gap decomposition for
Markov chains*. The experiment compares the multiscale
Metropolis-within-hit-and-run kernel $S$ with the ideal hit-and-run kernel
$\bar S$ for a synthetic Bayesian logistic-regression posterior.

For the coordinate test functions

$$
f_1(\beta)=\beta_1,
\qquad
f_k(\beta)=\beta_k,
$$

the reported quantities are

$$
\frac{\mathcal E_S(f_j)}{\mathcal E_{\bar S}(f_j)}
\quad\text{and}\quad
\frac{\mathcal E_S(f_j)}{\lVert f_j-\pi f_j\rVert_\pi^2},
\qquad j\in\{1,k\}.
$$

These coordinatewise comparisons illustrate Proposition 7.1; they are not
estimates of a worst-case spectral gap.

## Synthetic posterior

The prior is $\beta\sim N_k(0,I_k)$. Set $q_0=8$, let
$r=n/(2q_0)=n/16$, and define

$$
a_{\xi,r}=\sqrt{\frac{2(\xi-1)}{r}}.
$$

For each informed coordinate $h\in\{1,\ldots,8\}$, the data contain $r$
identical pairs of observations. Both rows of the pair are
$a_{\xi,r}e_h^\mathsf T$, while their responses are respectively $1$ and $0$.
All entries in columns $9,\ldots,k$ are zero. Consequently,

$$
I_k+\frac14\Xi^\mathsf T\Xi
=\mathrm{diag}(\xi I_8,I_{k-8}),
$$

so the negative log-posterior is $1$-strongly convex and $\xi$-smooth.
Coordinate $1$ is likelihood-informed, whereas coordinate $k$ retains its
prior-scale curvature.

The three paths use every power of two from $2^4$ through $2^{12}$:

| Path | Varied parameter | Fixed parameters |
| --- | --- | --- |
| Conditioning | $\xi\in\{16,32,\ldots,4096\}$ | $(k,n)=(16,128)$ |
| Dimension | $k\in\{16,32,\ldots,4096\}$ | $(\xi,n)=(8,128)$ |
| Sample size | $n\in\{16,32,\ldots,4096\}$ | $(\xi,k)=(8,16)$ |

## Stationary draws and transition kernels

The construction permits independent exact draws from $\pi$. The first eight
posterior coordinates are independent with marginal density proportional to

$$
t\longmapsto \phi(t)\mathrm{sech}^{2r}
\!\left(\frac{a_{\xi,r}t}{2}\right),
$$

where $\phi$ is the standard-normal density. The code proposes $t\sim N(0,1)$
and accepts it with probability
$\mathrm{sech}^{2r}(a_{\xi,r}t/2)$. The remaining coordinates are
independent $N(0,1)$ variables. This exact stationary sampler is available
because of the specially factorized design above; it is not a generic method
for logistic-regression posteriors.

For an $S$ update, the code draws a direction $w$ uniformly from the unit
sphere and sets $u=w^\mathsf T\beta$. It then draws

$$
s\sim\mathrm{Unif}\{0,\ldots,J-1\},
\qquad J=1+\lceil\log_2\xi\rceil,
$$

proposes $u'\sim N(u,2^s/\xi)$, and applies the Metropolis acceptance rule for
the conditional line density.

For a $\bar S$ update, the mode of the conditional line density is located by
60 bisection iterations. A proposal from $N(u_{\mathrm{mode}},1)$ is then
accepted by rejection sampling. The Gaussian envelope is valid analytically
because the line density is $1$-strongly log-concave. The output column
`maximum_log_envelope_error` records the largest sampled log-envelope
diagnostic; it is not a proof of the global envelope inequality. As in the
manuscript, “exact” here is subject to floating-point evaluation and the fixed
60-step mode calculation.

For every stationary state $\beta^{(i)}$, the $S$ and $\bar S$ transitions use
separate directions and separate transition randomness. Thus
$\beta_S^{(i)}$ and $\beta_{\bar S}^{(i)}$ are conditionally independent given
$\beta^{(i)}$.

## Installation

Python 3.12.13 was used to check the supplied results. Install the
three pinned direct dependencies from the package root:

```sh
python -m pip install -r requirements.txt
```

The plotting scripts select Matplotlib's noninteractive `Agg` backend, so they
do not require a display or a Tcl/Tk installation.

## Reproducing the outputs

To regenerate the six figures and path-specific fitted-power tables from the
supplied Monte Carlo CSV files, run:

```sh
python run_all_six_plots.py --plots-only
```

This is the fast route for checking figure construction. In the environment
described above it took about 2 seconds. It does not rerun the Monte Carlo
experiment. It rebuilds the figures, fitted-power tables, selected-ratio
tables, and the two consolidated CSV files in `results/` from the supplied
Monte Carlo CSV files. It leaves the raw Monte Carlo CSV files and configuration
files unchanged.

To rerun all 27 configurations with the manuscript setting of
$N=100{,}000$ independent stationary one-step experiments per configuration,
run:

```sh
python run_all_six_plots.py
```

The default batch size is 500. The $k$ path uses seed 8172028, and the $\xi$
and $n$ paths use seed 8172030, with NumPy's `Generator(PCG64)`. Retaining the
dependency versions, seeds, sample size, and batch size reproduces the supplied
random-number stream. The checked full run took about 115 seconds; runtime is
hardware-dependent. CSV values can differ at the last few floating-point
digits across platforms even when the substantive results agree.

Both commands overwrite the derived figures, fitted-power and selected-ratio
tables, and consolidated CSV files. A full run also replaces the path-specific
raw Monte Carlo CSV files and all three configuration files. Retain a clean
copy of the package if the supplied outputs are needed for comparison.

`code/section7_dirichlet_study.py` supplies the posterior, stationary sampler,
transition, estimator, and confidence-interval functions used by the two path
runners. It is a shared implementation module rather than the reproduction
entry point; use `run_all_six_plots.py` as shown above.

## Estimators and error bars

For a fixed coordinate function $f=f_j$ and replication $i$, define

$$
A_i=\frac12\{f(\beta^{(i)})-f(\beta_S^{(i)})\}^2,
\quad
B_i=\frac12\{f(\beta^{(i)})-f(\beta_{\bar S}^{(i)})\}^2,
\quad
C_i=f(\beta^{(i)})^2.
$$

Symmetry gives $\pi f_j=0$, so
$\lVert f_j-\pi f_j\rVert_\pi^2=\mathbb E_\pi[C_i]$. With bars denoting sample
means over $i=1,\ldots,N$, the plotted estimates are $\bar A/\bar B$ and
$\bar A/\bar C$.

For $\widehat R=\bar A/\bar B$, the estimated standard error on the log scale
is

$$
\widehat{\mathrm{se}}\{\log(\widehat R)\}
=\frac{\mathrm{sd}
\!\left(A_i/\bar A-B_i/\bar B\right)}{\sqrt N}.
$$

The pointwise 95% interval is

$$
\widehat R\exp\!\left[
\pm1.96\,\widehat{\mathrm{se}}\{\log(\widehat R)\}
\right].
$$

For $\bar A/\bar C$, replace $B_i$ by $C_i$. Pairing through the common
$\beta^{(i)}$ is retained in the influence values above. These intervals
measure Monte Carlo error and are pointwise, not simultaneous.

All figure axes are logarithmic. Reported power estimates are ordinary
least-squares slopes from regressing the natural logarithm of a ratio on the
natural logarithm of the path parameter, using all nine points. The slope is
unchanged if another logarithm base is used.

## Package layout

```text
.
├── README.md
├── requirements.txt
├── run_all_six_plots.py          # reproduction entry point
├── run_k_path.py                 # k path
├── run_xi_n_paths.py             # xi and n paths
├── code/
│   └── section7_dirichlet_study.py
├── figures/                      # six manuscript figure panels
├── results/                      # regenerated consolidated outputs
└── path_outputs/
    ├── k_path/
    │   ├── figures/
    │   └── results/
    └── xi_n_paths/
        ├── figures/
        └── results/
```

The figure filenames have the form
`endpoint_coordinates_<path>_<ratio>.png`, where `<path>` is `k`, `xi`, or
`n`. Files ending in `E_S_to_E_bar_S.png` show
$\mathcal E_S/\mathcal E_{\bar S}$; files ending in
`E_S_to_posterior_variance.png` show
$\mathcal E_S/\lVert f-\pi f\rVert_\pi^2$. Orange circles denote $j=1$, and
blue diamonds denote $j=k$ in every panel. The legend appears in the two
$\xi$-path panels and the same encoding is used in the $k$- and $n$-path
panels.

The detailed CSV files include the two estimated Dirichlet forms, the posterior
second moment, both ratios and intervals, acceptance diagnostics, and the
configuration variables. The fitted-power CSV files give the log-log slope and
$R^2$ for each path, ratio, and coordinate.

No software license is included in this archive.
