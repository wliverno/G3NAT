#!/usr/bin/env python3
"""Per-row orbital map for the Gaussian 6-31G(d,p) (Cartesian d) matrices in this dataset.

Row i of F and S belongs to atom atom_index[i] (0-based, PDB/Gaussian input order), shell
shell_index[i], and component component[i] (S, PX, PY, PZ, DXX, DYY, DZZ, DXY, DXZ, DYZ).
type_code is Gaussian's own per-function code (1000*l + component index) as printed in the
IBfTyp array of the matrix-element file, stored so the labels can be checked independently.

The dump's IBfAtm array is not used: it is corrupt (its early entries duplicate IBfTyp).
The independent SHELL TO ATOM MAP block is the reference for atom assignment instead.

What is checked where: export_matrices_hdf5.py applies its "verified" label only after
checking, per sequence, the element order (the dump's atomic numbers against the atoms
table) and the per-function type codes (derived type_code against the dump's IBfTyp). It
does not compare against the SHELL TO ATOM MAP; that shell-to-atom agreement is checked by
the unit test tests/test_dataset/test_gaussian_basis.py on one dump header (atatatatatat).
"""
import numpy as np

SHELLS = {"H": (0, 0, 1), "C": (0, -1, -1, 2), "N": (0, -1, -1, 2),
          "O": (0, -1, -1, 2), "P": (0, -1, -1, -1, 2)}
COMPONENTS = {0: ("S",), -1: ("S", "PX", "PY", "PZ"), 1: ("PX", "PY", "PZ"),
              2: ("DXX", "DYY", "DZZ", "DXY", "DXZ", "DYZ")}
TYPE_CODE = {"S": 1, "PX": 1001, "PY": 1002, "PZ": 1003, "DXX": 2001, "DYY": 2002,
             "DZZ": 2003, "DXY": 2004, "DXZ": 2005, "DYZ": 2006}
ELEMENT_OF_Z = {1: "H", 6: "C", 7: "N", 8: "O", 15: "P"}


def derive_basis(elements):
    atom, shell, stype, comp = [], [], [], []
    s = 0
    for a, el in enumerate(elements):
        if el not in SHELLS:
            raise ValueError(f"element {el!r} (atom {a}) has no basis definition")
        for t in SHELLS[el]:
            for c in COMPONENTS[t]:
                atom.append(a); shell.append(s); stype.append(t); comp.append(c)
            s += 1
    return {"atom_index": np.asarray(atom, np.int32),
            "shell_index": np.asarray(shell, np.int32),
            "shell_type": np.asarray(stype, np.int8),
            "component": np.asarray(comp, "S3"),
            "type_code": np.asarray([TYPE_CODE[c] for c in comp], np.int16)}


def parse_dump_header(path):
    out = {"atomic_numbers": [], "ibftyp": [], "shell_to_atom": [], "shell_types": []}
    block = None
    with open(path, "r", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.startswith("NAtoms="):
                out["nbasis"] = int(s.split("NBasis=")[1].split()[0])
            elif s.startswith("IAn="):
                out["atomic_numbers"] += [int(x) for x in s[4:].split()]
            elif s.startswith("IBfTyp="):
                out["ibftyp"] += [int(x) for x in s[7:].split()]
            elif s.startswith("Label"):
                if block == "shell_types":
                    break
                block = ("shell_to_atom" if "SHELL TO ATOM MAP" in s
                         else "shell_types" if "SHELL TYPES" in s else None)
            elif block and s.startswith("IArr="):
                out[block] += [int(x) for x in s[5:].split()]
    return out
