# Design-of-experiments methods for the training-configuration factorial

This document specifies, completely, how the training-configuration factorial is
analysed: the design, the responses, the statistical model, how the model is chosen,
how significance and multiplicity are handled, and what is reported. Anyone following
it should reproduce the analysis exactly; nothing in the pipeline is left to the
analyst's discretion at analysis time. All numbers quoted here were verified against
the reference implementation and its outputs (section 10) on 2026-08-09.

## 1. Factors and design

Three controllable factors and a replication factor:

- `alpha` in {0, 1}: onsite parameterization mode (0 = free context, 1 = per-base
  baseline).
- `b` in {0, 0.25, 0.5, 0.75, 1.0}: LDOS weight in the training loss
  `a*T_loss + b*LDOS_loss + (1-b)*DOS_loss`. Treated as a continuous factor with
  linear and quadratic components.
- `geom` in {0, 1}: X3DNA edge-geometry features off/on.
- seed: 3 initialization seeds per cell, treated as replication, not as a factor.

The primary design is the full cross: 2 x 5 x 2 = 20 cells, 3 seeds per cell,
60 runs, fully balanced.

Two additional `b` levels exist in the run archive and are excluded from the primary
analysis:

- `b = 0.9` (12 runs) is a held-out reserve slice, excluded by design. Its purpose
  is confirmatory: claims formed on the primary design can be tested against it
  afterwards. Folding it into the primary fit would destroy that function, so balance
  or sample size is never a reason to include it. It may be analysed only as an
  explicitly labelled replication pass (`factorial_unified.py b6-replication`), and
  results from that pass are reported as replication checks, never as headline
  findings.
- `b = 0.1` (9 runs) has empty (alpha, geom) cells and cannot join any balanced fit.

Every analysis states which `b` levels entered it, and the analysis scripts print the
levels found in their input as their first output line.

## 2. Responses

Eight responses are computed per run. Direction ("better") is fixed in advance per
response; where better is a judgment rather than a fact about the objective, the
response is marked DESCRIPTIVE and no optimum is claimed for it.

| response     | meaning                                                        | direction |
|--------------|----------------------------------------------------------------|-----------|
| `dos_t`      | DOS+T validation loss at best-val, unweighted                  | lower     |
| `ldos`       | LDOS disagreement                                              | lower     |
| `loc_gap`    | localization-gap bias (pred minus target); 0 is correct        | nearest 0 |
| `best_epoch` | epoch of best validation loss (ruggedness proxy)               | DESCRIPTIVE |
| `len_slope`  | log10 T decay per bp from the length sweep                     | DESCRIPTIVE |
| `len_r2`     | R2 of the exponential length fit                               | higher    |
| `sub_dos`    | substitution DOS skill vs the predict-no-change null           | higher    |
| `sub_t`      | substitution T skill vs the same null                          | higher    |

Circularity flag: `b` weights the LDOS term inside the training loss, so any effect
of `b` on the `ldos` response is circular by construction. It is always labelled as
such and never reported as a finding about the model.

## 3. The statistical model is enumerated, not inherited

The candidate term set is the complete set implied by the factor structure
alpha x (b, b^2) x geom -- every product of {alpha}, {b, b^2}, {geom} up to the
three-way -- which is 11 terms:

```
alpha, b, b^2, geom,
alpha:b, alpha:b^2, geom:alpha, geom:b, geom:b^2,
alpha:geom:b, alpha:geom:b^2
```

The term list is generated from the factors by code, and the design matrix is built
from that list, so the two cannot drift apart. Terms are dropped from the candidate
set only if the design cannot support them (rank deficiency, i.e. aliasing, or too
few residual degrees of freedom) -- never for being non-significant and never to
shrink the multiple-comparison family, since either would amount to choosing the
model from the answer.

On the primary design all 11 terms are estimable: the 12-column design matrix has
full rank, residual df 48, and condition number 66.6.

`b` is centred (`b_c = b - mean(b)`) before squaring and before forming
interactions. Uncentred, the polynomial and interaction columns are strongly
collinear with variance inflation factors of 25-50 (`b` 25.9, `b^2` 24.9,
`alpha:b` 49.7, `alpha:b^2` 37.7); centring removes most of this. Centring for
polynomial models is coding hygiene, not a modelling choice; a model that respects
effect hierarchy is invariant to it in fit (Peixoto 1990), but the per-term tests
and conditioning are not.

## 4. Per-response occupancy check before fitting

Full rank of the design matrix is necessary but not sufficient: a response may be
missing for some runs, and an interaction can then be formally estimable while being
supported by a single occupied cell, in which case its inclusion silently absorbs
variance from a region the design cannot inform. Therefore, before fitting any
response, the occupancy table for that response -- counts per (alpha, geom) x b cell,
computed over the runs where that response is finite -- is printed and inspected. An
interaction enters the fit only where both parents actually vary across the occupied
cells; otherwise the model is restricted (fewer terms, or a restricted factor range,
stated explicitly in the output).

On the current data all eight responses occupy all 60 runs, so the full 11-term
model applies everywhere; the check remains mandatory because response coverage has
differed from design coverage before.

## 5. Estimation and testing

Each response is fitted by ordinary least squares. Each non-intercept term is tested
by a nested drop-one F test: the full model against the model with that single term
removed, F = ((RSS_reduced - RSS_full)/df1) / (RSS_full/df2). Every reported row
carries n, df2, the F and p per term, and the residual SD of the full model. The
residual SD is the seed-level noise floor for that response and effect sizes are
read against it.

One model specification is applied to both response types (section 7) and all
responses within a type. Fitting different term sets to the same data in different
scripts is prohibited: a term can appear significant in one specification and vanish
in another purely because the other terms were not held constant, which makes
side-by-side quotation of such outputs meaningless.

## 6. Model reduction: per response, hierarchical, rule fixed in advance

After the full-model table is reported, each response is reduced by hierarchical
backward elimination with the rule fixed in advance:

1. Among the currently retained terms, consider only terms of the highest order
   present whose higher-order dependents have all been removed (effect hierarchy /
   marginality: a main effect is never removed while an interaction containing it
   remains; Nelder 1977, Peixoto 1990).
2. Remove the eligible term with the largest p, if that p exceeds 0.10.
3. Stop when every remaining eligible term has p <= 0.10.

The full-model table is always reported before reduction, so every elimination step
is auditable against it.

The unit of reduction is the individual response. A rule requiring a term to be
null in every response before dropping it retains everything by construction: with
8 responses, a genuinely null term has probability 1 - 0.9^8 = 57% of landing at
p <= 0.10 somewhere by chance alone. Each response is its own experiment and gets
its own reduced model; this is also what the per-response occupancy rule already
implies.

Retained models on the current data (from `factorial_reduced.out`):

| response     | retained terms                                              |
|--------------|-------------------------------------------------------------|
| `dos_t`      | alpha                                                       |
| `ldos`       | alpha, b, b^2, geom, alpha:b^2                              |
| `loc_gap`    | alpha, geom                                                 |
| `best_epoch` | 10 of 11 (all but alpha:geom:b)                             |
| `len_slope`  | alpha                                                       |
| `len_r2`     | alpha                                                       |
| `sub_dos`    | 10 of 11 (all but alpha:geom:b)                             |
| `sub_t`      | all 11, including the three-way                             |

The spread of these retained models is itself informative: no single fixed term set
is right for all responses. A single inherited 5-term model was simultaneously
over-specified for the fit and length responses (which want only alpha) and
under-specified for the substitution channels (which want nearly the full set), and
conclusions about factors tested under it moved when the specification did.

## 7. Location and dispersion, one model, both always

Two response types are analysed with the same model specification and the same
F-test machinery:

- LOCATION: one observation per run; y is the response itself. n = 60; df2 depends
  on the retained model (48 under the full 11-term model).
- DISPERSION: one observation per cell; y = max - min of the response across the 3
  seeds in that cell. n = 20. This answers "does a factor move the seed scatter",
  which is a distinct and independently reportable question.

The two are not comparable on p-values alone: dispersion has a fraction of the
power (n = 20 vs 60), so a null there is weak evidence, not evidence of absence.
n and df2 are printed on every row for exactly this reason.

The range of 3 seeds is a crude, right-skewed spread estimator. Dispersion is
therefore analysed on both the raw and log10 scales, with a Shapiro-Wilk normality
check (Shapiro and Wilk 1965) on the raw spreads, and a dispersion term is accepted
only if it is significant on both scales. Log-scale analysis of dispersion measures
in factorials follows Box and Meyer (1986). On the current data 6 of 8 responses
fail Shapiro-Wilk on the raw spreads, so the log row is the operative one, but the
both-scales rule is applied uniformly.

## 8. Multiplicity

All p-values from a given analysis run -- location, dispersion raw, and dispersion
log, over all responses and all tested terms -- form one family, and the
Benjamini-Hochberg procedure (Benjamini and Hochberg 1995) is applied across that
family at q = 0.05. The family size m is computed from the tests actually performed
in that run and is printed with the results; every claimed survivor is quoted
together with its m. Under the full 11-term model on the primary design,
m = 264. Comparing p-values across analyses with different family sizes is not
meaningful, which is one more reason a single unified analysis script is used.

## 9. The best-cell table is a first-class deliverable

"Which effects are significant" and "what settings should we use" are different
questions. Significance asks whether an average marginal effect can be resolved at
n = 3 seeds per cell; a practitioner choosing a configuration needs the best
observed cell, which exists for every response regardless of significance, and a
best-vs-worst cell contrast is a far larger difference than a marginal main effect
and routinely clears the seed noise floor when no single term does.

For every response with a defined direction, the analysis reports: the best cell
(alpha, b, geom), its per-seed range, the worst cell, the best-worst gap, and the
gap divided by the pooled within-cell SD of that response (gap/SD > 3: clears the
seed noise comfortably; 1-3: real but modest; < 1: the "best" cell is not
meaningfully better). Responses without a defined direction (`best_epoch`,
`len_slope`) appear marked DESCRIPTIVE with no optimum claimed.

On the current data (from `best_cell_table.py`): the largest contrasts are
`ldos` gap/SD = 8.6, `sub_dos` = 6.0, `sub_t` = 4.4, `dos_t` = 4.3. Across the six
responses with a defined optimum, alpha = 0 wins 4, geom = 0 wins 5, and b = 0.5
wins 3 (no other b level wins more than 1). A factor level that wins on most
responses is a defensible default even when its ANOVA term is not significant.

The standing guardrail travels with the table: a best-vs-worst contrast shows the
CELLS differ, not which factor caused it -- cells differ in more than one factor at a
time, and only the per-term F-test of section 5 holds the others constant. Best-cell
results select settings; they never establish effects.

## 10. Reference implementation

The analysis is implemented in three scripts (internal repository,
`G3NAT-internal/scratch/analysis/`):

- `factorial_unified.py` -- term enumeration, location and dispersion analyses,
  per-response hierarchical reduction, BH over the printed family; one file, so the
  location and dispersion models cannot drift apart. Primary output:
  `factorial_reduced.out` (full-model table, reduction trace, dispersion tables,
  Shapiro-Wilk results, BH family and survivors).
- `best_cell_table.py` -- the best/worst cell table and win tallies
  (`best_cell_table.out`).
- `design_estimability.py` -- design occupancy, rank/condition/df for candidate
  models, per-term VIF.

Where prose (including this document) and code disagree, the code and its committed
outputs are the source of truth.

## 11. Reporting requirements and known noise sources

Every analysis output includes: the factor levels that entered the fit, n and df2
per row, the residual SD per response, the BH family size, and explicit labels on
circular responses (section 2) and on any replication pass over the reserve slice
(section 1). Results caches are never used as the input of the script that produces
them; a rebuild is forced whenever the underlying run set may have changed.

Single-seed results are not treated as results, and neither are single-configuration
results: with 20 cells trained and evaluated, sweeping the full design costs only
inference time.

One known limitation of the replication structure: the batch sampler constructs an
unseeded random generator (`g3nat/training/utils.py`, lines 60 and 67), so runs are
not bit-reproducible even at fixed initialization seed, and the within-cell seed
spread contains uncontrolled batch-order noise in addition to initialization noise.
Dispersion analyses remain interpretable -- spread is spread, whatever its
decomposition -- but same-seed pairing across configurations is not a matched
comparison and is never analysed as one.

## References

- Benjamini, Y., Hochberg, Y. (1995). "Controlling the false discovery rate: a
  practical and powerful approach to multiple testing." *J. R. Statist. Soc. B*
  57(1), 289-300. doi:10.1111/j.2517-6161.1995.tb02031.x
- Box, G.E.P., Meyer, R.D. (1986). "Dispersion effects from fractional designs."
  *Technometrics* 28(1), 19-27.
- Nelder, J.A. (1977). "A reformulation of linear models." *J. R. Statist. Soc. A*
  140(1), 48-63. doi:10.2307/2344517
- Peixoto, J.L. (1990). "A property of well-formulated polynomial regression
  models." *The American Statistician* 44(1), 26-30.
  doi:10.1080/00031305.1990.10475687
- Shapiro, S.S., Wilk, M.B. (1965). "An analysis of variance test for normality
  (complete samples)." *Biometrika* 52(3/4), 591-611.
