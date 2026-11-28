# src/pose.py

import numpy as np
from scipy.spatial.transform import Rotation


IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


def dihedral(p0, p1, p2, p3):
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    b2_hat = b2 / np.linalg.norm(b2)
    x = np.dot(n1, n2)
    y = np.dot(np.cross(n1, n2), b2_hat)
    return np.arctan2(y, x)


def measure_tau(template, X, pair):
    quad = template["torsion_atoms"][pair]
    row = template["name_to_row"]
    return dihedral(X[row[quad[0]]],
                    X[row[quad[1]]],
                    X[row[quad[2]]],
                    X[row[quad[3]]])


def n_dof(template):
    return 6 + len(template["rotatable"])


def identity_pose(template):
    T = np.zeros(3)
    q = IDENTITY_QUAT.copy()
    dtau = np.zeros(len(template["rotatable"]))
    return T, q, dtau


def pose_to_coords(template, T, q, dtau):
    X = template["X0"].copy()
    row = template["name_to_row"]
    rotatable = template["rotatable"]

    for i in range(len(rotatable)):
        pair = rotatable[i]
        quad = template["torsion_atoms"][pair]
        proximal = quad[1]
        distal = quad[2]

        p_prox = X[row[proximal]]
        p_dist = X[row[distal]]

        axis = p_dist - p_prox
        axis = axis / np.linalg.norm(axis)
        rot = Rotation.from_rotvec(axis * dtau[i])

        for name in template["moving"][pair]:
            r = row[name]
            X[r] = p_prox + rot.apply(X[r] - p_prox)

    centroid = X.mean(axis=0)
    X = Rotation.from_quat(q).apply(X - centroid) + centroid + T
    return X


def rmsd(X, Y):
    diff = X - Y
    return float(np.sqrt((diff * diff).sum(axis=1).mean()))

def write_pose(X, source_pdb, out_path):
    lines = []
    i = 0
    for line in source_pdb.read_text().splitlines():
        if line.startswith("HETATM"):
            if line[76:78].strip().upper() == "H":
                continue
            x, y, z = X[i]
            line = line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:]
            i += 1
        lines.append(line)
    assert i == len(X), f"wrote {i} coords for {len(X)} atoms"
    out_path.write_text("\n".join(lines) + "\n")