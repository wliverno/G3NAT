# `readmat` -- Gaussian binary matrix-element file converter

`readmat` converts the binary matrix-element file that Gaussian 16 writes under
`output=(matrix,i4lab)` into plaintext, which `readMAT.m` then parses into the Fock and
overlap matrices. It is the first step of the transport pipeline after the DFT itself.

## THIRD-PARTY CODE -- NOT COVERED BY THE REPOSITORY'S MIT LICENCE

The `.F` files in this directory are **Gaussian Interface Code** (the `gauopen` toolkit),
distributed by Gaussian, Inc. under the **Gaussian Interface Code Open-Source Public License
v1.0** -- a modified Mozilla Public License 2.0. The full text is in `LICENSE.txt` here, and
is also published at <http://gaussian.com/public-licensev1.0>.

That licence permits redistribution in source or binary form, free or commercial. It also
requires (section 3.2) that if you distribute the compiled binary you must make the source
available. **A previous revision of this repository committed the compiled `readmat` binary
with no source, which did not satisfy that condition.** The binary has been removed and the
source added here instead.

Do not relicense these files or strip their notices (section 3.3). The rest of G3NAT is MIT;
this directory is not.

## What we changed

`readmat.F` is Gaussian's own `readfaf.F` example program with capacity increases for
systems larger than the example anticipated:

```
line 9:  MaxAt   3000 -> 30000        max atoms
line 9:  MaxArr  1e7  -> 1e9          max array elements
```

Nothing else differs. The program still self-identifies as `Program ReadFaf` internally.

### One repair applied here (2026-08-03)

The version recovered from `NEGFCode/gauopen/` also had `Call Close_MatF(...)` at line 166
where upstream has `Call Close_FAF(...)`. **`Close_MatF` does not exist in this generation of
the library** and the link fails with `undefined reference to close_matf_`. It is defined
only in the 2018 "v2" gauopen (`qcmatrixio.F`, 33 KB), which in turn lacks `Rd_HeadA`,
`Rd_ChBuf`, `DAOInts`, `Rd_SPA` and `QCM_MaxINZR` that this same file calls -- so the file as
found could not build against *either* library generation. Reverted to `Close_FAF`, which
links cleanly against the 47 KB (2023-12) `qcmatrixio.F` vendored alongside it.

Every `readmat.F` copy found elsewhere on the cluster carries the same `Close_MatF` call, so
this is a long-standing inconsistency rather than local corruption. The shipped binary
predates it or was built against a library nobody currently has co-located.

## Building -- TESTED 2026-08-03, all four flags are required

```bash
module load gcc/10.2.0
gfortran -g -O0 -ffixed-line-length-none -fallow-argument-mismatch -mcmodel=medium \
    -o readmat qcmatrixio.F qcmatrix.F readmat.F
```

None of these are optional, and a naive `gfortran -O2 -o readmat *.F` fails at every step:

| flag | why it is needed |
|---|---|
| `-ffixed-line-length-none` | Line 9 is **74 characters**. Fixed-form Fortran truncates at column 72, so the capacity edit above silently cuts the trailing comma off the `Parameter` statement and the parse collapses. |
| `-fallow-argument-mismatch` | `Rd_RInd` is called with both a `COMPLEX(8)` and a `REAL(8)` array. gfortran 10 promoted this from warning to error. |
| `-mcmodel=medium` | `MaxArr=1e9` of `Real*8` is ~8 GB of static BSS; the default small code model overflows with `relocation truncated to fit: R_X86_64_32S`. |
| `gcc/10.2.0` | The version the shipped binary was built with, recovered from its `DW_AT_producer` debug string. `gcc/11.2.0` is the current module default and is untested here. |

`-g -O0` also comes from that debug string. Optimisation is irrelevant for a utility that
does one pass of file I/O.

**VERIFIED 2026-08-10 (after the output-format patch below): a freshly built `readmat`
reproduces the working binary's output byte for byte on everything the pipeline reads.**
The earlier inconclusive test used `aaacgacg_gaussian.mat`, which neither binary could
read -- the file, not the build, was the problem. Regression on
`validation_L12/gggggggggggg/gggggggggggg.mat` against the working binary's shipped
`gggggggggggg.txt`:

- all 47 `Label` lines present and the `Label OVERLAP` header byte-identical;
- the `Label OVERLAP` and `Label ALPHA FOCK MATRIX` `RArr=` data streams are
  byte-identical (3,800,613 lines each, 0 differing);
- the parsed H0 is bit-identical to a same-session control parse of the working
  binary's own dump. (Cross-session H0 comparisons show a benign 1e-7 last-ASCII-digit
  scatter from BLAS library drift in the S^-1/2 eigendecomposition -- present even when
  reparsing the reference's own txt -- so bit-level comparisons must be same-session.)

**The as-vendored source was a DIFFERENT PRINTER REVISION from the working binary's
(lost) source**, and the 2026-08-05 "unlabelled numbers" failure was exactly that, not a
compile-flag problem: no `' Label '` prefix on block headers, no `'RArr='`/`'IArr='`
prefixes on array data, `NRI` as I2, `LenBuf` as I5, and a trailing `IType=` integer.
The last one is load-bearing: `readMAT.m` and `readmat_parse.py` size the matrix from
the LAST integer on the header line, so a trailing `IType=    0` silently yields 0x0
matrices. Patched in `readmat.F` (Formats 1110/1150/1160/1170 and the block-header
Write) to reproduce the working layout byte-for-byte; see the comment above Format 1110.

Two header-array quirks observed and deliberately NOT chased (the pipeline never reads
them): the patched build prints `IAtTyp=` as zeros where the working binary printed
constants (604/628 etc.), and the WORKING binary's `IBfAtm=` lines contain what appear
to be shell-type codes rather than a basis-to-atom map (the patched build's ascending
atom indices look more plausible). Anyone consuming those arrays should verify them
independently first.

The pipeline invokes it as `readmat <seq>.mat > <seq>.txt` -- see
`DNADataset/combined_script.slurm`.

## `MaxBf`: raised from 10000 to 30000 on 2026-08-05

`MaxBf` is a **hard compile-time ceiling** on basis functions, checked at `readmat.F:70-71`,
which prints a diagnostic and stops rather than producing bad output. It was left at the
upstream example's value of 10000, which was about to become a problem.

Measured basis-function density in this dataset is ~11.48 per atom at B3LYP/6-31G(d,p), and
structures run ~63.5 atoms per base pair:

| duplex length | atoms | basis functions | vs old 10000 | vs current 30000 |
|---|---|---|---|---|
| 8  | 507  | 5806 (measured)  | ok | ok |
| 12 | 762  | 8748 (measured)  | ok, 12.5% under | ok |
| 14 | ~889 | ~10200 (projected) | **REJECTED** | ok |
| 16 | ~1016 | ~11700 (projected) | **REJECTED** | ok |
| 20 | ~1270 | ~14600 (projected) | REJECTED | ok |
| 24 | ~1524 | ~17500 (projected) | REJECTED | ok |

The failure mode this avoids is expensive: the ceiling is checked at CONVERSION time, so a
14-bp run would have consumed a full Gaussian calculation (~8 h at L=12, more at L=14) and
then been rejected at the step that reads the result. Raising it costs only memory in this
small utility -- `IBfAtm` and `IBfTyp` are integer arrays, so 30000 costs a few hundred KB.

Note `MaxBf=30000` keeps line 9 at exactly 74 characters, so the
`-ffixed-line-length-none` requirement below is unchanged. If you raise it further, count
the digits: adding a character does not break the build given that flag, but the line is
already past the column-72 fixed-form limit and depends on it.

## Upstream

The full toolkit (including the C and Python interfaces, `qc.make`, and the format
documentation `binarfile.txt` / `interface.txt`) is available from Gaussian's interfacing
page. Only the three files needed to build `readmat` are vendored here.
