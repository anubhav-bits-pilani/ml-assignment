from __future__ import annotations

import pandas as pd


# takes whatever csv someone uploaded and lines it up with what the models expect.
# returns (X, y or None, notes for the user, error message or None)
def prepare(df: pd.DataFrame, meta, target_choice=None):
    notes = []
    feats = meta["feature_names"]
    # be forgiving about header case and stray spaces, e.g. " area " still matches Area
    canon = {c.strip().lower(): c for c in df.columns}
    rename = {canon[f.lower()]: f for f in feats if f.lower() in canon}
    df = df.rename(columns=rename)
    missing = [f for f in feats if f not in df.columns]
    if missing:
        return None, None, notes, f"The file is missing {len(missing)} feature column(s): {', '.join(missing)}"

    # target column: whatever the user picked, else anything that looks like "Class"
    target = target_choice
    if target is None:
        target = canon.get(meta["target"].lower())
    extra = [c for c in df.columns if c not in feats and c != target]
    if extra:
        notes.append(f"Ignoring {len(extra)} extra column(s): {', '.join(map(str, extra[:8]))}")

    # anything that isn't a number becomes NaN, and those rows get dropped below
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    bad_rows = X.isna().any(axis=1)
    if target is not None:
        y_raw = df[target].astype(str).str.strip().str.upper()
        bad_rows |= df[target].isna()
    if bad_rows.any():
        notes.append(f"Dropped {int(bad_rows.sum())} row(s) with missing / non-numeric values")
        X = X[~bad_rows]
    if len(X) == 0:
        return None, None, notes, "No usable rows left after cleaning."
    # no labels -> the app can still show predictions, just no metrics
    if target is None:
        return X.reset_index(drop=True), None, notes, None

    y_raw = y_raw[~bad_rows]
    # labels are matched case-insensitively; anything we never trained on is an error, not a guess
    lookup = {c.upper(): i for i, c in enumerate(meta["class_names"])}
    unknown = sorted(set(y_raw) - set(lookup))
    if unknown:
        return None, None, notes, (f"Target column has labels the models never saw: {unknown[:10]}. "
                                   f"Expected one of {meta['class_names']}")
    y = y_raw.map(lookup).to_numpy(dtype=int)
    return X.reset_index(drop=True), y, notes, None
