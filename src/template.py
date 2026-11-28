# src/template.py

from pathlib import Path

import numpy as np
import requests
import gemmi
import networkx as nx

from pose import dihedral


HERE = Path(__file__).parent
DATA = HERE.parent / "data"
PREP = DATA / "PREP"
CCD = DATA / "CCD"
CCD.mkdir(exist_ok=True)


def load_ligand_codes():
    codes = {}
    text = (DATA / "ads_codes.txt").read_text()
    for chunk in text.split(","):
        parts = chunk.split()
        if len(parts) != 2:
            continue
        codes[parts[0]] = parts[1]
    return codes


LIG = load_ligand_codes()


def read_lig(code):
    serial = []
    name = []
    element = []
    xyz = []

    for line in (PREP / f"{code}_ligand.pdb").read_text().splitlines():
        if not line.startswith("HETATM"):
            continue
        el = line[76:78].strip().upper()
        if el == "H":
            continue
        serial.append(int(line[6:11]))
        name.append(line[12:16].strip())
        element.append(el)
        xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])

    return {
        "serial": serial,
        "name": name,
        "element": element,
        "X0": np.array(xyz),
    }


def read_conect(code, keep):
    bonds = {}
    for s in keep:
        bonds[s] = set()

    for line in (PREP / f"{code}_ligand.pdb").read_text().splitlines():
        if not line.startswith("CONECT"):
            continue

        numbers = []
        for field in line.split()[1:]:
            numbers.append(int(field))

        owner = numbers[0]
        partners = numbers[1:]

        if owner not in bonds:
            continue
        for partner in partners:
            if partner not in bonds:
                continue
            bonds[owner].add(partner)
            bonds[partner].add(owner)

    return bonds


def fetch_ccd(lig_code):
    path = CCD / f"{lig_code}.cif"
    if path.exists():
        return path

    url = f"https://files.rcsb.org/ligands/download/{lig_code}.cif"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    text = response.text
    if not text.startswith("data_"):
        raise ValueError(f"{lig_code}: response is not a cif")

    path.write_text(text)
    return path


def parse_ccd(path):
    block = gemmi.cif.read(str(path)).sole_block()

    ccd_atoms = {}
    for row in block.find("_chem_comp_atom.", ["atom_id", "type_symbol"]):
        name = gemmi.cif.as_string(row[0])
        el = gemmi.cif.as_string(row[1]).upper()
        ccd_atoms[name] = {"element": el}

    ccd_bonds = {}
    for row in block.find("_chem_comp_bond.",
                          ["atom_id_1", "atom_id_2", "value_order", "pdbx_aromatic_flag"]):
        name_a = gemmi.cif.as_string(row[0])
        name_b = gemmi.cif.as_string(row[1])
        order = gemmi.cif.as_string(row[2])
        aromatic = gemmi.cif.as_string(row[3]) == "Y"
        ccd_bonds[frozenset({name_a, name_b})] = {"order": order, "aromatic": aromatic}

    return ccd_atoms, ccd_bonds


def build_template(code):
    lig_code = LIG[code]

    lig = read_lig(code)
    assert len(set(lig["name"])) == len(lig["name"]), f"{code}: duplicate atom names"

    serial_to_name = {}
    for i in range(len(lig["serial"])):
        serial_to_name[lig["serial"][i]] = lig["name"][i]

    name_to_row = {}
    for i in range(len(lig["name"])):
        name_to_row[lig["name"][i]] = i

    raw_bonds = read_conect(code, set(lig["serial"]))

    pdb_bonds = set()
    for a in raw_bonds:
        for b in raw_bonds[a]:
            pdb_bonds.add(frozenset({serial_to_name[a], serial_to_name[b]}))

    ccd_atoms, ccd_bonds = parse_ccd(fetch_ccd(lig_code))

    ccd_heavy_atoms = set()
    for name in ccd_atoms:
        if ccd_atoms[name]["element"] != "H":
            ccd_heavy_atoms.add(name)

    ccd_heavy_bonds = set()
    for pair in ccd_bonds:
        name_a, name_b = tuple(pair)
        if name_a in ccd_heavy_atoms and name_b in ccd_heavy_atoms:
            ccd_heavy_bonds.add(pair)

    warnings = []
    extra_atoms = set(lig["name"]) - set(ccd_atoms)
    if extra_atoms:
        warnings.append(f"atoms not in CCD: {sorted(extra_atoms)}")
    unmodelled = ccd_heavy_atoms - set(lig["name"])
    if unmodelled:
        warnings.append(f"{len(unmodelled)} heavy atoms unmodelled")
    if pdb_bonds - ccd_heavy_bonds:
        warnings.append(f"bonds not in CCD: {sorted(pdb_bonds - ccd_heavy_bonds)}")
    if ccd_heavy_bonds - pdb_bonds:
        warnings.append(f"CCD bonds missing from CONECT: {sorted(ccd_heavy_bonds - pdb_bonds)}")

    bond_orders = {}
    for pair in pdb_bonds:
        if pair in ccd_bonds:
            bond_orders[pair] = ccd_bonds[pair]

    G = nx.Graph()
    for name in lig["name"]:
        G.add_node(name)
    for pair in pdb_bonds:
        a, b = tuple(pair)
        G.add_edge(a, b)

    if not nx.is_connected(G):
        warnings.append("ligand graph is disconnected")

    bridges = set()
    for a, b in nx.bridges(G):
        bridges.add(frozenset({a, b}))

    rotatable = []
    for pair in sorted(bridges, key=sorted):
        if pair not in bond_orders:
            continue
        if bond_orders[pair]["order"] != "SING":
            continue
        a, b = tuple(pair)
        H = G.copy()
        H.remove_edge(a, b)
        parts = list(nx.connected_components(H))
        if len(parts[0]) < 2 or len(parts[1]) < 2:
            continue
        rotatable.append(pair)

    F = G.copy()
    for pair in rotatable:
        a, b = tuple(pair)
        F.remove_edge(a, b)
    fragments = []
    for component in nx.connected_components(F):
        fragments.append(set(component))

    root_fragment = max(fragments, key=len)
    root_atom = sorted(root_fragment)[0]

    X0 = lig["X0"]
    moving = {}
    torsion_atoms = {}
    tau0 = {}

    for pair in rotatable:
        a, b = tuple(pair)
        H = G.copy()
        H.remove_edge(a, b)
        for component in nx.connected_components(H):
            if root_atom not in component:
                moving[pair] = set(component)

        if a in moving[pair]:
            distal = a
            proximal = b
        else:
            distal = b
            proximal = a

        ref_prox = sorted(n for n in G[proximal] if n != distal)[0]
        ref_dist = sorted(n for n in G[distal] if n != proximal)[0]

        quad = (ref_prox, proximal, distal, ref_dist)
        torsion_atoms[pair] = quad
        tau0[pair] = dihedral(X0[name_to_row[quad[0]]],
                              X0[name_to_row[quad[1]]],
                              X0[name_to_row[quad[2]]],
                              X0[name_to_row[quad[3]]])

    return {
        "code": code,
        "lig_code": lig_code,
        "names": lig["name"],
        "elements": lig["element"],
        "X0": X0,
        "name_to_row": name_to_row,
        "bonds": pdb_bonds,
        "bond_orders": bond_orders,
        "rotatable": rotatable,
        "moving": moving,
        "torsion_atoms": torsion_atoms,
        "tau0": tau0,
        "root_atom": root_atom,
        "n_fragments": len(fragments),
        "warnings": warnings,
    }