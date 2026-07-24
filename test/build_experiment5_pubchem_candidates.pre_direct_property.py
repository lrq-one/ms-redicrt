from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
import sqlite3
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger


ROOT = Path.cwd().resolve()

TARGET_FP = (
    ROOT
    / "runs/experiments/experiment5_source_audit"
    / "retrieval_target_coverage.csv"
)

MOL_FP = (
    ROOT
    / "data/proc/nist20_qtof_cid_safe19659"
    / "mol_df.pkl"
)

LEGACY_FP = (
    ROOT
    / "../old/preproc_scripts/pubchem_ms2c"
    / "02_prepare_ms2c_candidates.py"
).resolve()

OLD_SRC = (
    ROOT
    / "../old/src"
).resolve()

OUT_DIR = (
    ROOT
    / "runs/experiments/molecular_retrieval"
    / "pubchem_legacy_full"
)

TARGET_OUT_DIR = OUT_DIR / "target_pools"

CACHE_DIR = (
    Path.home()
    / "datasets"
    / "pubchem_pugrest_10ppm_20260723"
)

MASS_CACHE_DIR = (
    CACHE_DIR
    / "mass_queries_by_connectivity"
)

PROPERTY_DB = (
    CACHE_DIR
    / "pubchem_properties.sqlite"
)

PPM = 10.0
MAX_CANDIDATES = 50
MORGAN_RADIUS = 2
PROPERTY_BATCH_SIZE = 500
NORMAL_REQUEST_DELAY = 0.25

for directory in [
    OUT_DIR,
    TARGET_OUT_DIR,
    CACHE_DIR,
    MASS_CACHE_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

for path in [
    TARGET_FP,
    MOL_FP,
    LEGACY_FP,
]:
    if not path.is_file():
        raise FileNotFoundError(path)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def atomic_text(
    path: Path,
    text: str,
) -> None:
    temporary = Path(
        str(path) + ".tmp"
    )

    temporary.write_text(
        text,
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def atomic_json(
    path: Path,
    payload: dict | list,
) -> None:
    atomic_text(
        path,
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
    )


def atomic_gzip_json(
    path: Path,
    payload: dict,
) -> None:
    temporary = Path(
        str(path) + ".tmp"
    )

    with gzip.open(
        temporary,
        "wt",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
        )

    os.replace(
        temporary,
        path,
    )


def chunks(
    values: list[int],
    size: int,
):
    for start in range(
        0,
        len(values),
        size,
    ):
        yield values[
            start:start + size
        ]


print("=" * 112)
print("LOAD LEGACY FILTER")
print("=" * 112)
print("LEGACY SOURCE:", LEGACY_FP)

sys.path.insert(
    0,
    str(OLD_SRC),
)

module_spec = (
    importlib.util.spec_from_file_location(
        "legacy_prepare_ms2c_candidates",
        LEGACY_FP,
    )
)

if (
    module_spec is None
    or module_spec.loader is None
):
    raise RuntimeError(
        f"Cannot import {LEGACY_FP}"
    )

legacy = (
    importlib.util.module_from_spec(
        module_spec
    )
)

module_spec.loader.exec_module(
    legacy
)

if not hasattr(
    legacy,
    "filter_candidates",
):
    raise RuntimeError(
        "filter_candidates was not found"
    )

RDLogger.DisableLog(
    "rdApp.*"
)


print()
print("=" * 112)
print("LOAD CURRENT TARGETS")
print("=" * 112)

coverage = pd.read_csv(
    TARGET_FP
)

mol_df = pd.read_pickle(
    MOL_FP
).copy()

coverage["target_mol_id"] = (
    pd.to_numeric(
        coverage["target_mol_id"],
        errors="raise",
    ).astype(int)
)

coverage["target_exact_mass"] = (
    pd.to_numeric(
        coverage["target_exact_mass"],
        errors="raise",
    ).astype(float)
)

mol_df["mol_id"] = (
    pd.to_numeric(
        mol_df["mol_id"],
        errors="raise",
    ).astype(int)
)

mol_df["exact_mw"] = (
    pd.to_numeric(
        mol_df["exact_mw"],
        errors="raise",
    ).astype(float)
)

membership = coverage.merge(
    mol_df[
        [
            "mol_id",
            "smiles",
            "inchikey_s",
            "formula",
            "exact_mw",
        ]
    ],
    left_on="target_mol_id",
    right_on="mol_id",
    how="left",
    validate="many_to_one",
)

if membership["smiles"].isna().any():
    missing = membership.loc[
        membership["smiles"].isna(),
        [
            "split",
            "target_mol_id",
            "target_connectivity_key",
        ],
    ]

    raise RuntimeError(
        "Targets missing from mol_df:\n"
        + missing.to_string(
            index=False
        )
    )

key_mismatch = (
    membership["target_connectivity_key"]
    .astype(str)
    != membership["inchikey_s"]
    .astype(str)
)

if key_mismatch.any():
    print(
        "WARNING: connectivity-key mismatches:",
        int(key_mismatch.sum()),
    )

membership[
    "exact_mass_abs_diff"
] = np.abs(
    membership["target_exact_mass"]
    - membership["exact_mw"]
)

split_memberships = (
    membership.groupby(
        "target_connectivity_key",
        sort=True,
    )["split"]
    .agg(
        lambda values: "|".join(
            sorted(
                set(
                    str(value)
                    for value in values
                )
            )
        )
    )
    .rename("split_memberships")
    .reset_index()
)

unique_targets = (
    membership.sort_values(
        [
            "target_connectivity_key",
            "target_mol_id",
            "split",
        ]
    )
    .drop_duplicates(
        "target_connectivity_key",
        keep="first",
    )
    .merge(
        split_memberships,
        on="target_connectivity_key",
        how="left",
        validate="one_to_one",
    )
    .reset_index(drop=True)
)

print("MEMBERSHIP ROWS       :", len(membership))
print("UNIQUE TARGETS        :", len(unique_targets))
print(
    "RANDOM TEST TARGETS   :",
    int(
        (
            membership["split"]
            == "random_test"
        ).sum()
    ),
)
print(
    "SCAFFOLD TEST TARGETS :",
    int(
        (
            membership["split"]
            == "scaffold_test"
        ).sum()
    ),
)
print(
    "MAX MASS DIFFERENCE   :",
    membership[
        "exact_mass_abs_diff"
    ].max(),
)

membership.to_csv(
    OUT_DIR
    / "target_split_membership.csv",
    index=False,
)

unique_targets[
    [
        "target_mol_id",
        "target_connectivity_key",
        "smiles",
        "formula",
        "exact_mw",
        "split_memberships",
    ]
].to_csv(
    OUT_DIR
    / "unique_targets.csv",
    index=False,
)


connection = sqlite3.connect(
    PROPERTY_DB,
    timeout=120,
)

connection.execute(
    """
    CREATE TABLE IF NOT EXISTS compound_property (
        cid INTEGER PRIMARY KEY,
        inchikey TEXT,
        smiles TEXT,
        formula TEXT,
        exact_mass REAL,
        fetched_at_utc TEXT
    )
    """
)

connection.execute(
    """
    CREATE INDEX IF NOT EXISTS
    compound_property_inchikey_idx
    ON compound_property(inchikey)
    """
)

connection.commit()


def request_json(
    request: urllib.request.Request,
    *,
    timeout: int = 300,
    retries: int = 8,
    allow_404: bool = False,
) -> dict:
    last_error = None

    for attempt in range(
        1,
        retries + 1,
    ):
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout,
            ) as response:
                raw = response.read()

            time.sleep(
                NORMAL_REQUEST_DELAY
            )

            return json.loads(
                raw.decode("utf-8")
            )

        except urllib.error.HTTPError as error:
            body = error.read().decode(
                "utf-8",
                errors="replace",
            )

            if (
                error.code == 404
                and allow_404
            ):
                return {
                    "_http_status": 404,
                    "_body": body[:1000],
                }

            last_error = (
                f"HTTP {error.code}: "
                f"{body[:500]}"
            )

            retry_after = 0

            try:
                retry_after = int(
                    error.headers.get(
                        "Retry-After",
                        "0",
                    )
                )
            except Exception:
                retry_after = 0

            if error.code not in {
                202,
                429,
                500,
                502,
                503,
                504,
            }:
                raise RuntimeError(
                    last_error
                ) from error

            wait_seconds = max(
                retry_after,
                min(
                    5 * attempt,
                    30,
                ),
            )

        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
        ) as error:
            last_error = repr(error)

            wait_seconds = min(
                5 * attempt,
                30,
            )

        print(
            f"    request retry "
            f"{attempt}/{retries}; "
            f"sleep={wait_seconds}s; "
            f"error={last_error}",
            flush=True,
        )

        time.sleep(
            wait_seconds
        )

    raise RuntimeError(
        "Request failed after retries: "
        f"{last_error}"
    )


def mass_cache_path(
    target_key: str,
    exact_mass: float,
) -> Path:
    safe_key = "".join(
        character
        for character in target_key
        if (
            character.isalnum()
            or character in "-_"
        )
    )

    return (
        MASS_CACHE_DIR
        / (
            f"{safe_key}_"
            f"{exact_mass:.6f}_"
            f"{PPM:g}ppm.json.gz"
        )
    )


def query_mass_cids(
    target_key: str,
    exact_mass: float,
) -> list[int]:
    cache_path = mass_cache_path(
        target_key,
        exact_mass,
    )

    if cache_path.is_file():
        with gzip.open(
            cache_path,
            "rt",
            encoding="utf-8",
        ) as handle:
            payload = json.load(
                handle
            )

        return sorted(
            {
                int(cid)
                for cid in payload[
                    "cids"
                ]
            }
        )

    tolerance = (
        exact_mass
        * PPM
        * 1e-6
    )

    lower = (
        exact_mass
        - tolerance
    )

    upper = (
        exact_mass
        + tolerance
    )

    url = (
        "https://pubchem.ncbi.nlm.nih.gov/"
        "rest/pug/compound/exact_mass/range/"
        f"{lower:.10f}/{upper:.10f}/"
        "cids/JSON"
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "lrq-ms2-retrieval/1.0"
            )
        },
    )

    payload = request_json(
        request,
        allow_404=True,
    )

    if payload.get(
        "_http_status"
    ) == 404:
        cids = []

    else:
        cids = (
            payload
            .get(
                "IdentifierList",
                {},
            )
            .get(
                "CID",
                [],
            )
        )

        waiting = payload.get(
            "Waiting",
            {},
        )

        list_key = (
            waiting.get("ListKey")
            or waiting.get(
                "ListKeyValue"
            )
        )

        if (
            not cids
            and list_key
        ):
            poll_url = (
                "https://pubchem.ncbi.nlm.nih.gov/"
                "rest/pug/compound/listkey/"
                f"{list_key}/cids/JSON"
            )

            for poll_index in range(
                1,
                61,
            ):
                time.sleep(2)

                poll_request = (
                    urllib.request.Request(
                        poll_url,
                        headers={
                            "User-Agent": (
                                "lrq-ms2-retrieval/1.0"
                            )
                        },
                    )
                )

                poll_payload = request_json(
                    poll_request,
                    retries=4,
                    allow_404=True,
                )

                cids = (
                    poll_payload
                    .get(
                        "IdentifierList",
                        {},
                    )
                    .get(
                        "CID",
                        [],
                    )
                )

                if cids:
                    break

                print(
                    f"    list-key poll "
                    f"{poll_index}/60",
                    flush=True,
                )

    cids = sorted(
        {
            int(cid)
            for cid in cids
        }
    )

    atomic_gzip_json(
        cache_path,
        {
            "target_connectivity_key": (
                target_key
            ),
            "target_exact_mass": float(
                exact_mass
            ),
            "ppm": PPM,
            "lower_mass": float(
                lower
            ),
            "upper_mass": float(
                upper
            ),
            "queried_at_utc": utc_now(),
            "cid_count": len(cids),
            "cids": cids,
        },
    )

    return cids


def existing_cids(
    cids: list[int],
) -> set[int]:
    present: set[int] = set()

    for batch in chunks(
        cids,
        800,
    ):
        placeholders = ",".join(
            "?"
            for _ in batch
        )

        rows = connection.execute(
            (
                "SELECT cid "
                "FROM compound_property "
                f"WHERE cid IN ({placeholders})"
            ),
            batch,
        ).fetchall()

        present.update(
            int(row[0])
            for row in rows
        )

    return present


def fetch_property_batch(
    cids: list[int],
) -> list[dict]:
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/"
        "rest/pug/compound/cid/property/"
        "SMILES,InChIKey,"
        "MolecularFormula,ExactMass/JSON"
    )

    body = urllib.parse.urlencode({
        "cid": ",".join(
            str(cid)
            for cid in cids
        )
    }).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "User-Agent": (
                "lrq-ms2-retrieval/1.0"
            ),
            "Content-Type": (
                "application/"
                "x-www-form-urlencoded"
            ),
        },
    )

    payload = request_json(
        request,
    )

    return (
        payload
        .get(
            "PropertyTable",
            {},
        )
        .get(
            "Properties",
            [],
        )
    )


def insert_properties(
    properties: list[dict],
) -> set[int]:
    rows = []
    received: set[int] = set()

    for item in properties:
        cid = item.get("CID")

        if cid is None:
            continue

        cid = int(cid)

        smiles = (
            item.get("SMILES")
            or item.get(
                "ConnectivitySMILES"
            )
            or item.get(
                "CanonicalSMILES"
            )
            or ""
        )

        exact_mass = item.get(
            "ExactMass"
        )

        try:
            exact_mass_value = (
                float(exact_mass)
                if exact_mass is not None
                else None
            )
        except (
            TypeError,
            ValueError,
        ):
            exact_mass_value = None

        rows.append((
            cid,
            str(
                item.get(
                    "InChIKey",
                    "",
                )
            ),
            str(smiles),
            str(
                item.get(
                    "MolecularFormula",
                    "",
                )
            ),
            exact_mass_value,
            utc_now(),
        ))

        received.add(cid)

    if rows:
        connection.executemany(
            """
            INSERT OR REPLACE INTO
            compound_property(
                cid,
                inchikey,
                smiles,
                formula,
                exact_mass,
                fetched_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

        connection.commit()

    return received


def ensure_properties(
    cids: list[int],
) -> tuple[int, int]:
    present_before = existing_cids(
        cids
    )

    missing = [
        cid
        for cid in cids
        if cid not in present_before
    ]

    if missing:
        total_batches = (
            len(missing)
            + PROPERTY_BATCH_SIZE
            - 1
        ) // PROPERTY_BATCH_SIZE

        for batch_index, batch in enumerate(
            chunks(
                missing,
                PROPERTY_BATCH_SIZE,
            ),
            start=1,
        ):
            properties = (
                fetch_property_batch(
                    batch
                )
            )

            received = insert_properties(
                properties
            )

            print(
                f"    properties "
                f"{batch_index}/{total_batches}: "
                f"requested={len(batch)} "
                f"received={len(received)}",
                flush=True,
            )

    present_after = existing_cids(
        cids
    )

    still_missing = [
        cid
        for cid in cids
        if cid not in present_after
    ]

    if still_missing:
        print(
            "    retrying unresolved "
            f"property CIDs: "
            f"{len(still_missing)}",
            flush=True,
        )

        for batch in chunks(
            still_missing,
            100,
        ):
            properties = (
                fetch_property_batch(
                    batch
                )
            )

            insert_properties(
                properties
            )

    final_present = existing_cids(
        cids
    )

    return (
        len(present_before),
        len(final_present),
    )


def load_property_rows(
    cids: list[int],
) -> list[tuple]:
    rows_by_cid: dict[int, tuple] = {}

    for batch in chunks(
        cids,
        800,
    ):
        placeholders = ",".join(
            "?"
            for _ in batch
        )

        rows = connection.execute(
            (
                "SELECT "
                "cid, inchikey, smiles, "
                "formula, exact_mass "
                "FROM compound_property "
                f"WHERE cid IN ({placeholders}) "
                "ORDER BY cid ASC"
            ),
            batch,
        ).fetchall()

        for row in rows:
            rows_by_cid[
                int(row[0])
            ] = row

    return [
        rows_by_cid[cid]
        for cid in sorted(
            rows_by_cid
        )
    ]


def pool_paths(
    target_key: str,
) -> tuple[
    Path,
    Path,
    Path,
]:
    return (
        TARGET_OUT_DIR
        / f"{target_key}.pkl.gz",
        TARGET_OUT_DIR
        / f"{target_key}.csv.gz",
        TARGET_OUT_DIR
        / f"{target_key}.json",
    )


def save_candidate_pool(
    candidate_df: pd.DataFrame,
    pickle_path: Path,
    csv_path: Path,
) -> None:
    temporary_pickle = Path(
        str(pickle_path) + ".tmp"
    )

    candidate_df.to_pickle(
        temporary_pickle,
        compression="gzip",
    )

    os.replace(
        temporary_pickle,
        pickle_path,
    )

    temporary_csv = Path(
        str(csv_path) + ".tmp"
    )

    candidate_df.drop(
        columns=["mol"],
        errors="ignore",
    ).to_csv(
        temporary_csv,
        index=False,
        compression="gzip",
    )

    os.replace(
        temporary_csv,
        csv_path,
    )


def validate_cached_pool(
    pickle_path: Path,
    metadata_path: Path,
) -> dict | None:
    if not (
        pickle_path.is_file()
        and metadata_path.is_file()
    ):
        return None

    try:
        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        candidate_df = pd.read_pickle(
            pickle_path
        )

        if (
            len(candidate_df)
            == int(
                metadata[
                    "legacy_filtered_count"
                ]
            )
            and int(
                candidate_df[
                    "is_true_structure"
                ].sum()
            )
            == int(
                metadata[
                    "true_structure_count"
                ]
            )
        ):
            metadata[
                "status"
            ] = "cached"

            return metadata

    except Exception:
        return None

    return None


def build_target_pool(
    row: pd.Series,
) -> dict:
    target_mol_id = int(
        row["target_mol_id"]
    )

    target_key = str(
        row[
            "target_connectivity_key"
        ]
    )

    target_smiles = str(
        row["smiles"]
    )

    target_mass = float(
        row["exact_mw"]
    )

    pickle_path, csv_path, metadata_path = (
        pool_paths(
            target_key
        )
    )

    cached = validate_cached_pool(
        pickle_path,
        metadata_path,
    )

    if cached is not None:
        return cached

    cids = query_mass_cids(
        target_key,
        target_mass,
    )

    if not cids:
        metadata = {
            "status": "no_pubchem_hits",
            "target_mol_id": (
                target_mol_id
            ),
            "target_connectivity_key": (
                target_key
            ),
            "target_exact_mass": (
                target_mass
            ),
            "split_memberships": str(
                row["split_memberships"]
            ),
            "raw_cid_count": 0,
            "complete_property_count": 0,
            "legacy_filtered_count": 0,
            "true_structure_count": 0,
            "true_structure_rank": None,
            "ready_50": 0,
            "finished_at_utc": utc_now(),
        }

        atomic_json(
            metadata_path,
            metadata,
        )

        return metadata

    cache_before, cache_after = (
        ensure_properties(
            cids
        )
    )

    property_rows = load_property_rows(
        cids
    )

    complete_rows = [
        row_value
        for row_value in property_rows
        if (
            row_value[1]
            and row_value[2]
            and row_value[3]
            and row_value[4] is not None
        )
    ]

    complete_property_count = len(
        complete_rows
    )

    (
        returned_mol_id,
        returned_smiles,
        candidate_df,
    ) = legacy.filter_candidates(
        list(complete_rows),
        target_smiles,
        target_mol_id,
        MAX_CANDIDATES,
        MORGAN_RADIUS,
    )

    candidate_df = (
        candidate_df.copy()
        .reset_index(drop=True)
    )

    candidate_df.insert(
        0,
        "generation_rank",
        np.arange(
            1,
            len(candidate_df) + 1,
            dtype=int,
        ),
    )

    candidate_df.insert(
        0,
        "target_connectivity_key",
        target_key,
    )

    candidate_df.insert(
        0,
        "target_mol_id",
        target_mol_id,
    )

    candidate_df.insert(
        0,
        "candidate_pool_id",
        target_key,
    )

    if (
        "inchikey_s"
        in candidate_df.columns
    ):
        candidate_keys = (
            candidate_df[
                "inchikey_s"
            ]
            .astype(str)
        )

    elif (
        "inchikey"
        in candidate_df.columns
    ):
        candidate_keys = (
            candidate_df[
                "inchikey"
            ]
            .astype(str)
            .str.split("-")
            .str[0]
        )

    else:
        candidate_keys = pd.Series(
            "",
            index=candidate_df.index,
        )

    candidate_df[
        "candidate_connectivity_key"
    ] = candidate_keys

    candidate_df[
        "is_true_structure"
    ] = (
        candidate_keys
        == target_key
    ).astype(int)

    true_rows = candidate_df[
        candidate_df[
            "is_true_structure"
        ] == 1
    ]

    true_count = int(
        len(true_rows)
    )

    true_rank = (
        int(
            true_rows[
                "generation_rank"
            ].iloc[0]
        )
        if true_count
        else None
    )

    negative_df = candidate_df[
        candidate_df[
            "is_true_structure"
        ] == 0
    ]

    if (
        len(negative_df)
        and "tanimoto"
        in negative_df.columns
    ):
        negative_tanimoto = (
            pd.to_numeric(
                negative_df[
                    "tanimoto"
                ],
                errors="coerce",
            )
            .dropna()
            .to_numpy(
                dtype=float
            )
        )
    else:
        negative_tanimoto = (
            np.asarray(
                [],
                dtype=float,
            )
        )

    ready_50 = int(
        len(candidate_df)
        == MAX_CANDIDATES
        and true_count == 1
    )

    save_candidate_pool(
        candidate_df,
        pickle_path,
        csv_path,
    )

    metadata = {
        "status": "complete",
        "target_mol_id": (
            target_mol_id
        ),
        "target_connectivity_key": (
            target_key
        ),
        "target_exact_mass": (
            target_mass
        ),
        "split_memberships": str(
            row["split_memberships"]
        ),
        "raw_cid_count": len(cids),
        "property_cache_before": (
            cache_before
        ),
        "property_cache_after": (
            cache_after
        ),
        "complete_property_count": (
            complete_property_count
        ),
        "legacy_filtered_count": int(
            len(candidate_df)
        ),
        "true_structure_count": (
            true_count
        ),
        "true_structure_rank": (
            true_rank
        ),
        "top_negative_tanimoto": (
            float(
                np.max(
                    negative_tanimoto
                )
            )
            if len(
                negative_tanimoto
            )
            else None
        ),
        "mean_negative_tanimoto": (
            float(
                np.mean(
                    negative_tanimoto
                )
            )
            if len(
                negative_tanimoto
            )
            else None
        ),
        "median_negative_tanimoto": (
            float(
                np.median(
                    negative_tanimoto
                )
            )
            if len(
                negative_tanimoto
            )
            else None
        ),
        "min_negative_tanimoto": (
            float(
                np.min(
                    negative_tanimoto
                )
            )
            if len(
                negative_tanimoto
            )
            else None
        ),
        "ready_50": ready_50,
        "pickle_path": str(
            pickle_path
        ),
        "csv_path": str(
            csv_path
        ),
        "finished_at_utc": utc_now(),
    }

    atomic_json(
        metadata_path,
        metadata,
    )

    return metadata


manifest = {
    "created_at_utc": utc_now(),
    "protocol_name": (
        "PubChem current PUG REST "
        "10ppm Morgan-R2 top50"
    ),
    "target_source": str(
        TARGET_FP
    ),
    "target_source_sha256": (
        sha256_file(
            TARGET_FP
        )
    ),
    "molecule_source": str(
        MOL_FP
    ),
    "molecule_source_sha256": (
        sha256_file(
            MOL_FP
        )
    ),
    "legacy_filter_source": str(
        LEGACY_FP
    ),
    "legacy_filter_sha256": (
        sha256_file(
            LEGACY_FP
        )
    ),
    "script_path": str(
        Path(__file__).resolve()
    ),
    "script_sha256": (
        sha256_file(
            Path(__file__).resolve()
        )
    ),
    "unique_target_count": int(
        len(unique_targets)
    ),
    "split_membership_count": int(
        len(membership)
    ),
    "ppm": PPM,
    "mass_definition": (
        "neutral exact molecular mass "
        "from current mol_df exact_mw"
    ),
    "fingerprint": "Morgan radius 2",
    "ranking": (
        "Tanimoto descending using "
        "legacy filter_candidates"
    ),
    "maximum_candidate_count": (
        MAX_CANDIDATES
    ),
    "maximum_includes_true_target": True,
    "target_injection": (
        "legacy filter appends the "
        "target before filtering and "
        "connectivity deduplication"
    ),
    "pre_filter_database_row_order": (
        "PubChem CID ascending for "
        "deterministic input ordering"
    ),
    "property_cache": str(
        PROPERTY_DB
    ),
    "mass_query_cache": str(
        MASS_CACHE_DIR
    ),
}

atomic_json(
    OUT_DIR
    / "manifest.json",
    manifest,
)


print()
print("=" * 112)
print("BUILD FULL CANDIDATE POOLS")
print("=" * 112)

summary_records: list[dict] = []

total_targets = len(
    unique_targets
)

for target_index, row in (
    unique_targets.iterrows()
):
    target_number = (
        target_index + 1
    )

    target_key = str(
        row[
            "target_connectivity_key"
        ]
    )

    print()
    print("-" * 112)
    print(
        f"[{target_number}/"
        f"{total_targets}] "
        f"mol_id="
        f"{int(row['target_mol_id'])} "
        f"key={target_key} "
        f"mass="
        f"{float(row['exact_mw']):.6f} "
        f"splits="
        f"{row['split_memberships']}"
    )
    print("-" * 112)

    result = None
    final_error = None

    for target_attempt in range(
        1,
        4,
    ):
        try:
            result = build_target_pool(
                row
            )

            break

        except Exception as error:
            final_error = repr(error)

            print(
                f"  TARGET ATTEMPT "
                f"{target_attempt}/3 "
                f"FAILED: "
                f"{final_error}",
                flush=True,
            )

            traceback.print_exc()

            time.sleep(
                15 * target_attempt
            )

    if result is None:
        result = {
            "status": "failed",
            "target_mol_id": int(
                row["target_mol_id"]
            ),
            "target_connectivity_key": (
                target_key
            ),
            "target_exact_mass": float(
                row["exact_mw"]
            ),
            "split_memberships": str(
                row[
                    "split_memberships"
                ]
            ),
            "raw_cid_count": None,
            "complete_property_count": (
                None
            ),
            "legacy_filtered_count": (
                None
            ),
            "true_structure_count": None,
            "true_structure_rank": None,
            "ready_50": 0,
            "error": final_error,
            "finished_at_utc": utc_now(),
        }

    summary_records.append(
        result
    )

    progress_df = pd.DataFrame(
        summary_records
    )

    temporary_progress = (
        OUT_DIR
        / "progress_summary.csv.tmp"
    )

    progress_df.to_csv(
        temporary_progress,
        index=False,
    )

    os.replace(
        temporary_progress,
        OUT_DIR
        / "progress_summary.csv",
    )

    print(
        "  status="
        f"{result.get('status')} "
        "raw="
        f"{result.get('raw_cid_count')} "
        "filtered="
        f"{result.get('legacy_filtered_count')} "
        "true_count="
        f"{result.get('true_structure_count')} "
        "ready_50="
        f"{result.get('ready_50')}",
        flush=True,
    )


connection.close()

summary_df = (
    pd.DataFrame(
        summary_records
    )
    .sort_values(
        "target_connectivity_key"
    )
    .reset_index(drop=True)
)

summary_df.to_csv(
    OUT_DIR
    / "candidate_pool_summary.csv",
    index=False,
)

candidate_frames = []

for _, summary_row in (
    summary_df.iterrows()
):
    target_key = str(
        summary_row[
            "target_connectivity_key"
        ]
    )

    pickle_path, _, _ = pool_paths(
        target_key
    )

    if pickle_path.is_file():
        candidate_frames.append(
            pd.read_pickle(
                pickle_path
            )
        )

if candidate_frames:
    all_candidates = pd.concat(
        candidate_frames,
        ignore_index=True,
    )
else:
    all_candidates = pd.DataFrame()

all_candidates.to_pickle(
    OUT_DIR
    / "all_unique_candidate_pools.pkl.gz",
    compression="gzip",
)

all_candidates.drop(
    columns=["mol"],
    errors="ignore",
).to_csv(
    OUT_DIR
    / "all_unique_candidate_pools.csv.gz",
    index=False,
    compression="gzip",
)

ready_count = int(
    pd.to_numeric(
        summary_df["ready_50"],
        errors="coerce",
    )
    .fillna(0)
    .sum()
)

failed_count = int(
    (
        summary_df["status"]
        .astype(str)
        == "failed"
    ).sum()
)

incomplete_count = int(
    len(summary_df)
    - ready_count
)

coverage_report = {
    "finished_at_utc": utc_now(),
    "unique_target_count": int(
        len(summary_df)
    ),
    "split_membership_count": int(
        len(membership)
    ),
    "ready_50_count": ready_count,
    "ready_50_fraction": (
        ready_count
        / len(summary_df)
        if len(summary_df)
        else None
    ),
    "incomplete_count": (
        incomplete_count
    ),
    "failed_count": failed_count,
    "candidate_row_count": int(
        len(all_candidates)
    ),
    "expected_candidate_rows_if_full": (
        int(
            len(summary_df)
            * MAX_CANDIDATES
        )
    ),
    "property_database": str(
        PROPERTY_DB
    ),
    "property_database_size_bytes": (
        PROPERTY_DB.stat().st_size
        if PROPERTY_DB.is_file()
        else 0
    ),
}

atomic_json(
    OUT_DIR
    / "coverage_report.json",
    coverage_report,
)

print()
print("=" * 112)
print("FINAL FULL-POOL REPORT")
print("=" * 112)
print(
    json.dumps(
        coverage_report,
        indent=2,
        ensure_ascii=False,
    )
)

print()
print(
    "SUMMARY:",
    OUT_DIR
    / "candidate_pool_summary.csv",
)
print(
    "POOLS:",
    OUT_DIR
    / "all_unique_candidate_pools.pkl.gz",
)
print(
    "MAPPING:",
    OUT_DIR
    / "target_split_membership.csv",
)
print(
    "MANIFEST:",
    OUT_DIR
    / "manifest.json",
)

if (
    ready_count
    == len(summary_df)
    and failed_count == 0
):
    print()
    print(
        "PUBCHEM_LEGACY_FULL_"
        "CANDIDATE_BUILD_COMPLETE"
    )
else:
    print()
    print(
        "PUBCHEM_LEGACY_FULL_"
        "CANDIDATE_BUILD_INCOMPLETE"
    )

    print(
        "Re-run the same command to "
        "resume failed or incomplete "
        "targets."
    )

    raise SystemExit(2)
