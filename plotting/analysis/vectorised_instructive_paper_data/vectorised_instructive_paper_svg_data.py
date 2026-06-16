import re
import json
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

SVG_FILE = "s41586-026-10190-7-2_with_ndnf.svg"   # <- change to your SVG path

OUT_NPZ = "all_curves_decr_incr_error_with_ndnf.npz"
OUT_H5  = "all_curves_decr_incr_error_with_ndnf.h5"
MAKE_PLOT = True

# ---------------- SVG parsing helpers ----------------
num_re = r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?'
token_re = re.compile(rf'([MmLlHhVvZz])|({num_re})')

def parse_path_points(d: str) -> np.ndarray:
    """Parse SVG path 'd' for basic polyline-like commands: M/L/H/V/Z (absolute/relative)."""
    tokens = token_re.findall(d.replace(',', ' '))
    seq = []
    for cmd, num in tokens:
        if cmd:
            seq.append(cmd)
        elif num:
            seq.append(float(num))

    pts = []
    x = y = 0.0
    start = None
    cmd = None

    def add_point(xx, yy):
        pts.append((xx, yy))

    i = 0
    while i < len(seq):
        if isinstance(seq[i], str):
            cmd = seq[i]
            i += 1

        if cmd in ('M', 'm'):
            is_rel = (cmd == 'm')
            first = True
            while i + 1 < len(seq) and not isinstance(seq[i], str) and not isinstance(seq[i + 1], str):
                dx, dy = seq[i], seq[i + 1]
                i += 2
                if is_rel:
                    x += dx; y += dy
                else:
                    x, y = dx, dy
                if start is None:
                    start = (x, y)
                add_point(x, y)
                if first:
                    first = False
                    cmd = 'l' if is_rel else 'L'  # implicit lineto

        elif cmd in ('L', 'l'):
            is_rel = (cmd == 'l')
            while i + 1 < len(seq) and not isinstance(seq[i], str) and not isinstance(seq[i + 1], str):
                dx, dy = seq[i], seq[i + 1]
                i += 2
                if is_rel:
                    x += dx; y += dy
                else:
                    x, y = dx, dy
                add_point(x, y)

        elif cmd in ('H', 'h'):
            is_rel = (cmd == 'h')
            while i < len(seq) and not isinstance(seq[i], str):
                val = seq[i]; i += 1
                x = x + val if is_rel else val
                add_point(x, y)

        elif cmd in ('V', 'v'):
            is_rel = (cmd == 'v')
            while i < len(seq) and not isinstance(seq[i], str):
                val = seq[i]; i += 1
                y = y + val if is_rel else val
                add_point(x, y)

        elif cmd in ('Z', 'z'):
            if start is not None:
                x, y = start
                add_point(x, y)

        else:
            raise ValueError(f"Unsupported SVG path command: {cmd}")

    return np.array(pts, dtype=float)

def mat_translate(tx, ty):
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1]], dtype=float)

def mat_matrix(a, b, c, d, e, f):
    # SVG matrix(a b c d e f):
    # [a c e]
    # [b d f]
    # [0 0 1]
    return np.array([[a, c, e],
                     [b, d, f],
                     [0, 0, 1]], dtype=float)

def parse_transform_str(s: str) -> np.ndarray:
    if not s:
        return np.eye(3, dtype=float)

    pattern = re.compile(r'(matrix|translate)\s*\(([^)]*)\)')
    M = np.eye(3, dtype=float)

    for kind, params in pattern.findall(s.strip()):
        nums = [float(x) for x in re.findall(num_re, params)]
        if kind == 'translate':
            tx = nums[0]
            ty = nums[1] if len(nums) > 1 else 0.0
            T = mat_translate(tx, ty)
        elif kind == 'matrix':
            T = mat_matrix(*nums)
        else:
            raise ValueError(kind)

        # apply in listed order
        M = M @ T

    return M

def apply_transform(points: np.ndarray, M: np.ndarray) -> np.ndarray:
    pts_h = np.c_[points, np.ones(len(points))]
    return (pts_h @ M.T)[:, :2]

def rescale(v, vmin, vmax, newmin, newmax):
    return (v - vmin) / (vmax - vmin) * (newmax - newmin) + newmin

# ---------------- Load SVG + locate paths ----------------
tree = ET.parse(SVG_FILE)
root = tree.getroot()

SVG_NS = "http://www.w3.org/2000/svg"
ns = {"svg": SVG_NS}

paths = root.findall(".//svg:path", ns)
if not paths:
    paths = root.findall(".//path")

parent_map = {c: p for p in root.iter() for c in p}

def find_path_by_id(pid: str):
    for p in paths:
        if p.attrib.get("id") == pid:
            return p
    raise KeyError(f"Could not find {pid} in SVG.")

def get_total_transform(el) -> np.ndarray:
    chain = []
    cur = el
    while True:
        par = parent_map.get(cur)
        if par is None:
            break
        chain.append(par)
        cur = par

    M = np.eye(3, dtype=float)
    for anc in reversed(chain):  # outer -> inner
        t = anc.attrib.get("transform")
        if t:
            M = M @ parse_transform_str(t)
    return M

def extract_path(pid: str) -> np.ndarray:
    el = find_path_by_id(pid)
    pts = parse_path_points(el.attrib["d"])
    pts = apply_transform(pts, get_total_transform(el))
    pts[:, 1] *= -1  # invert Y (SVG y increases downward)
    return pts

# ---------------- Define your two conditions ----------------
conditions = [
    dict(
        name="decreasing_error",
        x_range=(-1.5, 1.25),
        p_minus_path="path618",  # blue
        p_plus_path="path614",   # red
    ),
    dict(
        name="increasing_error",
        x_range=(-1.35123966942, 1.5),
        p_minus_path="path5230", # blue
        p_plus_path="path5226",  # red
    ),
    dict(
        name="ndnf_decreasing_error",
        x_range=(-1.40415632754, 1.5),
        p_minus_path="path7546",  # blue
        p_plus_path="path7542",  # red
    ),
    dict(
        name="ndnf_increasing_error",
        x_range=(-1.5, 1.5),
        p_minus_path="path6762",  # blue
        p_plus_path="path6758",  # red
    ),
]

# ---------------- Extract + rescale (shared scaling within each condition) ----------------
out = {}         # arrays for NPZ/H5
meta = {"svg_file": SVG_FILE, "conditions": []}

for cond in conditions:
    name = cond["name"]
    x_new_min, x_new_max = cond["x_range"]

    raw_pminus = extract_path(cond["p_minus_path"])
    raw_pplus  = extract_path(cond["p_plus_path"])

    # shared min/max within this condition to preserve relative offsets between p- and p+
    x_all = np.concatenate([raw_pminus[:, 0], raw_pplus[:, 0]])
    y_all = np.concatenate([raw_pminus[:, 1], raw_pplus[:, 1]])
    xmin, xmax = float(x_all.min()), float(x_all.max())
    ymin, ymax = float(y_all.min()), float(y_all.max())

    pminus = np.column_stack([
        rescale(raw_pminus[:, 0], xmin, xmax, x_new_min, x_new_max),
        rescale(raw_pminus[:, 1], ymin, ymax, 0.0, 1.0),
    ])
    pplus = np.column_stack([
        rescale(raw_pplus[:, 0], xmin, xmax, x_new_min, x_new_max),
        rescale(raw_pplus[:, 1], ymin, ymax, 0.0, 1.0),
    ])

    # store with clear keys: condition / p_minus, p_plus
    out[f"{name}__p_minus"] = pminus
    out[f"{name}__p_plus"]  = pplus

    meta["conditions"].append({
        "name": name,
        "x_range": [x_new_min, x_new_max],
        "y_range": [0.0, 1.0],
        "p_minus": {"label": "p-", "color": "blue", "path_id": cond["p_minus_path"]},
        "p_plus":  {"label": "p+", "color": "red",  "path_id": cond["p_plus_path"]},
        "shared_raw_bounds": {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax},
    })

# ---------------- Save NPZ (always) ----------------
# store metadata as JSON string inside the NPZ too
out_with_meta = dict(out)
out_with_meta["__meta_json__"] = np.array(json.dumps(meta), dtype=object)
np.savez(OUT_NPZ, **out_with_meta)
print(f"Saved NPZ: {OUT_NPZ}")

# ---------------- Save H5 (if available) ----------------
try:
    import h5py
    with h5py.File(OUT_H5, "w") as f:
        f.attrs["meta_json"] = json.dumps(meta)
        for cond in meta["conditions"]:
            g = f.create_group(cond["name"])
            g.create_dataset("p_minus", data=out[f"{cond['name']}__p_minus"])
            g.create_dataset("p_plus",  data=out[f"{cond['name']}__p_plus"])
            g.attrs["x_range"] = cond["x_range"]
            g.attrs["y_range"] = cond["y_range"]
            g.attrs["p_minus_path_id"] = cond["p_minus"]["path_id"]
            g.attrs["p_plus_path_id"] = cond["p_plus"]["path_id"]
    print(f"Saved H5:  {OUT_H5}")
except ImportError:
    print("h5py not installed; skipped H5 save.")

# ---------------- Optional sanity plot ----------------
if MAKE_PLOT:
    for cond in meta["conditions"]:
        plt.figure(figsize=(4, 5))
        name = cond["name"]
        pminus = out[f"{name}__p_minus"]
        pplus  = out[f"{name}__p_plus"]
        plt.plot(pminus[:, 0], pminus[:, 1], color="blue", label=f"{name}: p-")
        plt.plot(pplus[:, 0],  pplus[:, 1],  color="red",  label=f"{name}: p+")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
    plt.show()