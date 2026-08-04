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

**Not yet verified: that a freshly built `readmat` reproduces the shipped binary's output
byte for byte.** Both binaries were run against `aaacgacg_gaussian.mat` and both returned
after one line without reading it, with different status codes (`IU=0` from the fresh build,
`IU=-1` from the shipped one). That needs checking against a matrix-element file known to be
readable before this build is trusted for new records.

The pipeline invokes it as `readmat <seq>.mat > <seq>.txt` -- see
`DNADataset/combined_script.slurm`.

## KNOWN LIMIT: `MaxBf = 10000` basis functions

This was left at the example's value and is a **hard compile-time ceiling**, checked at
`readmat.F:70-71`, which prints a diagnostic and stops rather than producing bad output.

Measured basis-function density in this dataset is ~11.48 per atom at B3LYP/6-31G(d,p),
and structures run ~63.5 atoms per base pair:

| duplex length | atoms | basis functions | status |
|---|---|---|---|
| 8  | 507  | 5806 (measured)  | ok |
| 12 | 762  | 8748 (measured)  | ok, 12.5% under the ceiling |
| 14 | ~889 | ~10200 (projected) | **exceeds MaxBf** |
| 16 | ~1016 | ~11700 (projected) | **exceeds MaxBf** |

So anything at 14 base pairs or longer needs `MaxBf` raised and `readmat` rebuilt *before*
the DFT is run -- otherwise the conversion rejects a calculation that has already consumed
several hours of Gaussian time. Raising it costs only memory in this small utility.

## Upstream

The full toolkit (including the C and Python interfaces, `qc.make`, and the format
documentation `binarfile.txt` / `interface.txt`) is available from Gaussian's interfacing
page. Only the three files needed to build `readmat` are vendored here.
