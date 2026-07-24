#!/usr/bin/env python3

from __future__ import annotations

import argparse
from copy import deepcopy
import io
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cairosvg
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


# ============================================================
# Project import
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "src"))

from ms2spectra.utils import frag_utils  # noqa: E402


# ============================================================
# Appearance
# ============================================================

ACTIVE_COLOR = "#000000"
GHOST_COLOR = "#D0D3D6"
GHOST_OPACITY = "1"

SVG_WIDTH = 620
SVG_HEIGHT = 370
PNG_WIDTH = 620
PNG_HEIGHT = 370


# ============================================================
# Arguments
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export individual molecular fragment tiles from a real cached "
            "D3 atom-subset DAG."
        )
    )

    parser.add_argument(
        "--proc-dp",
        type=Path,
        default=Path("data/proc/nist20_qtof_cid_safe19659"),
    )

    parser.add_argument(
        "--frag-dp",
        type=Path,
        default=Path(
            "data/frag/"
            "nist20_qtof_cid_safe19659_d3_mhp_qtof_cid_nl_v1/"
            "dags"
        ),
    )

    parser.add_argument(
        "--mol-id",
        type=int,
        default=14647,
    )

    parser.add_argument(
        "--out-dp",
        type=Path,
        default=Path("figure/figures"),
    )

    parser.add_argument(
        "--per-depth",
        nargs=4,
        type=int,
        default=(1, 5, 5, 4),
        metavar=("D0", "D1", "D2", "D3"),
        help="Number of exported nodes at Depth 0, 1, 2 and 3.",
    )

    parser.add_argument(
        "--min-heavy-atoms",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--contact-cols",
        type=int,
        default=4,
    )

    return parser.parse_args()


# ============================================================
# Load molecule and cache
# ============================================================

def load_molecule(proc_dp: Path, mol_id: int) -> tuple[Chem.Mol, dict]:
    mol_fp = proc_dp / "mol_df.pkl"

    if not mol_fp.exists():
        raise FileNotFoundError(mol_fp)

    mol_df = pd.read_pickle(mol_fp)

    hit = mol_df[
        mol_df["mol_id"].astype(str) == str(mol_id)
    ]

    if hit.empty:
        raise KeyError(
            f"mol_id={mol_id} not found in {mol_fp}"
        )

    row = hit.iloc[0]

    mol = row.get("mol", None)

    if not isinstance(mol, Chem.Mol):
        smiles = None

        for column in [
            "smiles",
            "cano_smiles",
            "canonical_smiles",
            "mol_smiles",
        ]:
            if column in row.index and pd.notna(row[column]):
                smiles = str(row[column])
                break

        if smiles is None:
            raise RuntimeError(
                "No RDKit molecule or SMILES found in mol_df row."
            )

        mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise RuntimeError(
            f"Could not build molecule for mol_id={mol_id}"
        )

    mol = Chem.Mol(mol)

    # Compute precursor coordinates once.
    # All fragment tiles reuse these exact coordinates.
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(mol)

    return mol, row.to_dict()


def load_fragment_cache(
    frag_dp: Path,
    mol_id: int,
) -> dict:
    compressed_fp = frag_dp / f"{mol_id}.pickle.bz2"
    plain_fp = frag_dp / f"{mol_id}.pickle"

    if compressed_fp.exists():
        compressed = True
    elif plain_fp.exists():
        compressed = False
    else:
        raise FileNotFoundError(
            f"No cache found for mol_id={mol_id} in {frag_dp}"
        )

    return frag_utils.load_frag_d(
        str(mol_id),
        str(frag_dp),
        is_compressed=compressed,
    )


# ============================================================
# Read the real PyG cached atom-subset masks
# ============================================================

def extract_cached_nodes(
    frag_d: dict,
    num_atoms: int,
) -> tuple[np.ndarray, np.ndarray]:
    if "dag" not in frag_d:
        raise KeyError("frag_d does not contain 'dag'")

    dag = frag_d["dag"]

    if not hasattr(dag, "x"):
        raise TypeError(
            "frag_d['dag'] is missing PyG node tensor dag.x"
        )

    if not hasattr(dag, "node_feat_idxs"):
        raise TypeError(
            "frag_d['dag'] is missing node_feat_idxs"
        )

    node_feat_idxs = dag.node_feat_idxs

    if node_feat_idxs.ndim == 2:
        node_feat_idxs = node_feat_idxs[0]

    # The connected-component atom mask is stored in the "cc"
    # section of the PyG node feature tensor.
    cc_long = frag_utils.get_node_feats(
        dag.x,
        node_feat_idxs,
        "cc",
    ).long()

    node_masks = (
        frag_utils.th_long_to_mask(cc_long)
        .detach()
        .cpu()
        .numpy()
        .astype(bool)
    )

    node_masks = node_masks[:, :num_atoms]

    if "nodes_min_depth" not in frag_d:
        raise KeyError(
            "frag_d does not contain nodes_min_depth"
        )

    node_depths = np.asarray(
        frag_d["nodes_min_depth"],
        dtype=int,
    )

    if len(node_masks) != len(node_depths):
        raise RuntimeError(
            "Node mask count and depth count do not match: "
            f"{len(node_masks)} vs {len(node_depths)}"
        )

    return node_masks, node_depths


# ============================================================
# Select representative real nodes
# ============================================================

def jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 0.0

    return float(
        np.logical_and(mask_a, mask_b).sum() / union
    )


def select_diverse_nodes(
    candidates: list[int],
    limit: int,
    masks: np.ndarray,
    heavy_counts: np.ndarray,
) -> list[int]:
    if limit <= 0 or not candidates:
        return []

    remaining = list(dict.fromkeys(candidates))
    selected: list[int] = []

    # Start from the largest fragment.
    remaining.sort(
        key=lambda idx: (
            -int(heavy_counts[idx]),
            int(idx),
        )
    )

    selected.append(remaining.pop(0))

    while remaining and len(selected) < limit:
        best_idx = None
        best_score = -1e9

        for idx in remaining:
            similarity = max(
                jaccard(masks[idx], masks[chosen])
                for chosen in selected
            )

            diversity = 1.0 - similarity

            # Prefer visible fragments while also keeping structural diversity.
            size_score = min(
                float(heavy_counts[idx]) / 8.0,
                1.0,
            )

            score = 0.72 * diversity + 0.28 * size_score

            if score > best_score:
                best_score = score
                best_idx = idx

        assert best_idx is not None

        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


def select_nodes_by_depth(
    mol: Chem.Mol,
    masks: np.ndarray,
    depths: np.ndarray,
    per_depth: tuple[int, int, int, int],
    min_heavy_atoms: int,
) -> dict[int, list[int]]:
    is_heavy = np.asarray(
        [
            atom.GetAtomicNum() > 1
            for atom in mol.GetAtoms()
        ],
        dtype=bool,
    )

    heavy_counts = masks[:, is_heavy].sum(axis=1)

    selected: dict[int, list[int]] = {}

    for depth in range(4):
        candidates = np.where(depths == depth)[0].tolist()

        if depth == 0:
            candidates.sort(
                key=lambda idx: (
                    -int(heavy_counts[idx]),
                    int(idx),
                )
            )

            selected[depth] = candidates[:1]
            continue

        visible = [
            idx
            for idx in candidates
            if int(heavy_counts[idx]) >= min_heavy_atoms
        ]

        if visible:
            candidates = visible

        selected[depth] = select_diverse_nodes(
            candidates,
            per_depth[depth],
            masks,
            heavy_counts,
        )

    return selected


# ============================================================
# RDKit SVG rendering
# ============================================================

def create_precursor_svg(
    mol: Chem.Mol,
) -> str:
    drawer = rdMolDraw2D.MolDraw2DSVG(
        SVG_WIDTH,
        SVG_HEIGHT,
    )

    options = drawer.drawOptions()
    options.clearBackground = True
    options.padding = 0.055
    options.bondLineWidth = 2.0
    options.useBWAtomPalette()

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    return drawer.GetDrawingText()


def update_style_value(
    style: str,
    key: str,
    value: str,
) -> str:
    pattern = rf"{re.escape(key)}:[^;]+"

    if re.search(pattern, style):
        return re.sub(
            pattern,
            f"{key}:{value}",
            style,
        )

    return f"{style.rstrip(';')};{key}:{value}"


def _replace_svg_color(
    element,
    color: str,
) -> None:
    """
    Force every visible SVG primitive to one opaque color.

    This is deliberately applied to all paths first, including RDKit paths
    that do not carry atom-* or bond-* CSS classes. Those unclassified paths
    caused the black corner/cap artifacts in the previous renderer.
    """
    style = element.attrib.get("style", "")

    if style:
        style = update_style_value(style, "stroke", color)
        style = update_style_value(style, "stroke-opacity", "1")

        compact = style.replace(" ", "").lower()
        if "fill:" in compact and "fill:none" not in compact:
            style = update_style_value(style, "fill", color)
            style = update_style_value(style, "fill-opacity", "1")

        element.set("style", style)

    if "stroke" in element.attrib:
        old_stroke = element.attrib.get("stroke", "").lower()
        if old_stroke not in ("none", ""):
            element.set("stroke", color)
            element.set("stroke-opacity", "1")

    if "fill" in element.attrib:
        old_fill = element.attrib.get("fill", "").lower()
        if old_fill not in ("none", "", "#ffffff", "white"):
            element.set("fill", color)
            element.set("fill-opacity", "1")

    element.attrib.pop("opacity", None)


def _is_visible_primitive(element) -> bool:
    tag = element.tag.split("}")[-1].lower()
    return tag in {
        "path",
        "line",
        "polyline",
        "polygon",
        "circle",
        "ellipse",
        "text",
    }


def recolor_svg_by_mask(
    precursor_svg: str,
    mol: Chem.Mol,
    keep_mask: np.ndarray,
) -> str:
    """
    Render one cached atom-subset node using two SVG layers:

    base layer:
        every molecular primitive is uniformly pale gray

    retained overlay:
        only atom-label paths and bond paths belonging to the cached
        fragment mask are copied and painted black

    Because the whole SVG is ghosted before any classification, unclassified
    RDKit bond caps and aromatic endpoint paths cannot remain black.
    """
    root = ET.fromstring(precursor_svg)

    atom_pattern = re.compile(r"atom-(\d+)")
    bond_pattern = re.compile(r"bond-(\d+)")

    # Freeze original elements before appending the black overlay.
    original_elements = list(root.iter())
    black_overlay = []

    # ------------------------------------------------------------
    # Layer 1: ghost every visible molecular primitive.
    # This includes paths without atom/bond CSS classes.
    # ------------------------------------------------------------
    for element in original_elements:
        if _is_visible_primitive(element):
            _replace_svg_color(
                element,
                GHOST_COLOR,
            )

    # ------------------------------------------------------------
    # Layer 2: recover only cached retained atoms and bonds.
    # ------------------------------------------------------------
    for element in original_elements:
        class_name = element.attrib.get("class", "")

        bond_match = bond_pattern.search(class_name)
        atom_matches = atom_pattern.findall(class_name)

        retained = False

        if bond_match:
            bond_idx = int(bond_match.group(1))
            bond = mol.GetBondWithIdx(bond_idx)

            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()

            retained = bool(
                keep_mask[begin_idx]
                and keep_mask[end_idx]
            )

        elif atom_matches:
            atom_idx = int(atom_matches[0])
            retained = bool(keep_mask[atom_idx])

        if not retained:
            continue

        copied = deepcopy(element)
        _replace_svg_color(
            copied,
            ACTIVE_COLOR,
        )
        black_overlay.append(copied)

    # Retained structures are drawn after the ghost layer.
    for element in black_overlay:
        root.append(element)

    # Explicit white canvas.
    root.set("style", "background-color:#FFFFFF")

    return ET.tostring(
        root,
        encoding="unicode",
    )


def write_tile(
    svg_text: str,
    svg_fp: Path,
    png_fp: Path,
) -> None:
    svg_fp.write_text(
        svg_text,
        encoding="utf-8",
    )

    cairosvg.svg2png(
        bytestring=svg_text.encode("utf-8"),
        write_to=str(png_fp),
        output_width=PNG_WIDTH,
        output_height=PNG_HEIGHT,
        background_color="#FFFFFF",
    )


# ============================================================
# Contact sheet
# ============================================================

def create_contact_sheet(
    records: list[dict],
    output_fp: Path,
    columns: int,
) -> None:
    if not records:
        return

    thumb_width = 330
    thumb_height = 198
    label_height = 40
    margin = 24

    rows = math.ceil(len(records) / columns)

    canvas_width = (
        columns * thumb_width
        + (columns + 1) * margin
    )

    canvas_height = (
        rows * (thumb_height + label_height)
        + (rows + 1) * margin
    )

    canvas = Image.new(
        "RGB",
        (canvas_width, canvas_height),
        "white",
    )

    draw = ImageDraw.Draw(canvas)

    for position, record in enumerate(records):
        row_idx = position // columns
        col_idx = position % columns

        x = (
            margin
            + col_idx * (thumb_width + margin)
        )

        y = (
            margin
            + row_idx
            * (thumb_height + label_height + margin)
        )

        rgba = Image.open(
            record["png_fp"]
        ).convert("RGBA")

        white_background = Image.new(
            "RGBA",
            rgba.size,
            (255, 255, 255, 255),
        )

        white_background.alpha_composite(rgba)
        image = white_background.convert("RGB")

        image.thumbnail(
            (thumb_width, thumb_height),
            Image.Resampling.LANCZOS,
        )

        paste_x = x + (thumb_width - image.width) // 2
        paste_y = y + (thumb_height - image.height) // 2

        canvas.paste(
            image,
            (paste_x, paste_y),
        )

        label = (
            f"Depth {record['depth']}  "
            f"node {record['node_id']}"
        )

        draw.text(
            (x + 8, y + thumb_height + 8),
            label,
            fill="black",
        )

    canvas.save(
        output_fp,
        dpi=(300, 300),
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    proc_dp = args.proc_dp.resolve()
    frag_dp = args.frag_dp.resolve()

    output_dp = (
        args.out_dp.resolve()
        / f"fragment_tiles_mol_{args.mol_id}"
    )

    if output_dp.exists():
        for existing in output_dp.iterdir():
            if existing.is_file():
                existing.unlink()
    else:
        output_dp.mkdir(
            parents=True,
            exist_ok=True,
        )

    mol, row = load_molecule(
        proc_dp,
        args.mol_id,
    )

    frag_d = load_fragment_cache(
        frag_dp,
        args.mol_id,
    )

    masks, depths = extract_cached_nodes(
        frag_d,
        mol.GetNumAtoms(),
    )

    selected = select_nodes_by_depth(
        mol,
        masks,
        depths,
        tuple(args.per_depth),
        args.min_heavy_atoms,
    )

    precursor_svg = create_precursor_svg(mol)

    records: list[dict] = []

    running_idx = 0

    for depth in range(4):
        for node_idx in selected.get(depth, []):
            svg_text = recolor_svg_by_mask(
                precursor_svg,
                mol,
                masks[node_idx],
            )

            stem = (
                f"depth{depth}_"
                f"node{node_idx}_"
                f"{running_idx:02d}"
            )

            svg_fp = output_dp / f"{stem}.svg"
            png_fp = output_dp / f"{stem}.png"

            write_tile(
                svg_text,
                svg_fp,
                png_fp,
            )

            records.append(
                {
                    "depth": int(depth),
                    "node_id": int(node_idx),
                    "atoms": np.where(
                        masks[node_idx]
                    )[0].astype(int).tolist(),
                    "svg_fp": str(svg_fp),
                    "png_fp": str(png_fp),
                }
            )

            running_idx += 1

    create_contact_sheet(
        records,
        output_dp / "contact_sheet.png",
        args.contact_cols,
    )

    manifest = {
        "mol_id": int(args.mol_id),
        "smiles": (
            Chem.MolToSmiles(mol)
            if mol is not None
            else None
        ),
        "reached_depth": int(
            frag_d.get("reached_depth", -1)
        ),
        "selected_nodes": [
            {
                "depth": record["depth"],
                "node_id": record["node_id"],
                "atoms": record["atoms"],
                "png": Path(
                    record["png_fp"]
                ).name,
                "svg": Path(
                    record["svg_fp"]
                ).name,
            }
            for record in records
        ],
    }

    (
        output_dp / "selection.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("===== CACHE STRUCTURE =====")
    print("mol_id:", args.mol_id)
    print("num atoms:", mol.GetNumAtoms())
    print("cached nodes:", len(masks))
    print(
        "nodes by depth:",
        {
            depth: int(
                np.count_nonzero(depths == depth)
            )
            for depth in range(4)
        },
    )

    print()
    print("===== EXPORTED =====")

    for depth in range(4):
        print(
            f"Depth {depth}:",
            len(selected.get(depth, [])),
            selected.get(depth, []),
        )

    print()
    print("output:", output_dp)

    for fp in sorted(output_dp.iterdir()):
        if fp.is_file():
            print(fp.name)


if __name__ == "__main__":
    main()
