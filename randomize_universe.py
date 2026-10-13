"""
Randomize rows 50/50 into treatment/control, run balance checks on zip5, age, sex,
and (only when --export is passed) write:
  1. <stem>_full_universe.csv  - all rows + `universe` column
  2. <stem>_treatment.csv      - treatment rows only

Usage:
  python randomize_universe.py input.csv            # assign + balance report only
  python randomize_universe.py input.csv --export   # also write the two files (same draw
                                                    #   that was checked; skipped if flagged)
  python randomize_universe.py input.csv --export --force   # write even if flagged
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ZIP_MIN_COUNT = 10  # zips with fewer rows are pooled into "OTHER" for the chi-square test
SMD_FLAG = 0.1
ALPHA = 0.05


def assign(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    labels = np.array(["treatment"] * ((n + 1) // 2) + ["control"] * (n // 2))
    np.random.default_rng().shuffle(labels)
    out = df.copy()
    out["universe"] = labels
    return out


def smd_continuous(t, c):
    pooled = np.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2)
    return (t.mean() - c.mean()) / pooled if pooled > 0 else 0.0


def smd_binary(p1, p2):
    pooled = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    return (p1 - p2) / pooled if pooled > 0 else 0.0


def balance_report(df: pd.DataFrame) -> bool:
    t = df[df.universe == "treatment"]
    c = df[df.universe == "control"]
    ok = True
    print(f"\nRows: {len(df):,}  treatment={len(t):,}  control={len(c):,}\n")

    # --- Age ---
    ta = pd.to_numeric(t["age"], errors="coerce").dropna()
    ca = pd.to_numeric(c["age"], errors="coerce").dropna()
    tt = stats.ttest_ind(ta, ca, equal_var=False)
    ks = stats.ks_2samp(ta, ca)
    smd = smd_continuous(ta, ca)
    print("AGE")
    print(f"  mean   T={ta.mean():.2f}  C={ca.mean():.2f}")
    print(f"  median T={ta.median():.1f}  C={ca.median():.1f}")
    print(f"  missing T={t['age'].isna().sum()}  C={c['age'].isna().sum()}")
    print(f"  Welch t-test p={tt.pvalue:.3f} | KS p={ks.pvalue:.3f} | SMD={smd:+.3f}")
    if tt.pvalue < ALPHA or ks.pvalue < ALPHA or abs(smd) > SMD_FLAG:
        print("  ** FLAG"); ok = False

    # --- Sex ---
    sex = df["sex"].fillna("MISSING").astype(str).str.strip()
    ct = pd.crosstab(sex, df["universe"])
    chi = stats.chi2_contingency(ct)
    print("\nSEX")
    props = ct / ct.sum()
    for lvl in ct.index:
        p_t, p_c = props.loc[lvl, "treatment"], props.loc[lvl, "control"]
        s = smd_binary(p_t, p_c)
        flag = "  ** FLAG" if abs(s) > SMD_FLAG else ""
        print(f"  {lvl:<10} T={p_t:.1%}  C={p_c:.1%}  SMD={s:+.3f}{flag}")
        if flag: ok = False
    print(f"  chi-square p={chi.pvalue:.3f}")
    if chi.pvalue < ALPHA:
        print("  ** FLAG"); ok = False

    # --- Zip5 ---
    z = df["zip5"].fillna("MISSING").astype(str).str.strip().str.zfill(5)
    counts = z.value_counts()
    z_pooled = z.where(z.map(counts) >= ZIP_MIN_COUNT, "OTHER")
    ctz = pd.crosstab(z_pooled, df["universe"])
    chiz = stats.chi2_contingency(ctz)
    propz = ctz / ctz.sum()
    smds = propz.apply(lambda r: smd_binary(r["treatment"], r["control"]), axis=1)
    worst = smds.abs().sort_values(ascending=False).head(5)
    print("\nZIP5")
    print(f"  distinct zips={counts.size}  tested categories={len(ctz)} "
          f"(zips with <{ZIP_MIN_COUNT} rows pooled as OTHER)")
    print(f"  chi-square p={chiz.pvalue:.3f}  max |SMD|={smds.abs().max():.3f}")
    print("  largest gaps:")
    for zz in worst.index:
        print(f"    {zz}: T={propz.loc[zz,'treatment']:.2%}  C={propz.loc[zz,'control']:.2%}  "
              f"SMD={smds[zz]:+.3f}")
    if chiz.pvalue < ALPHA or (smds.abs() > SMD_FLAG).any():
        print("  ** FLAG"); ok = False

    print("\nRESULT:", "BALANCED - ok to export" if ok else "IMBALANCE FLAGGED - consider re-randomizing")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--export", action="store_true", help="write files if balance checks pass")
    ap.add_argument("--force", action="store_true", help="with --export, write even if flagged")
    ap.add_argument("--zip-col", default="zip5", help="name of the zip column (default zip5)")
    ap.add_argument("--age-col", default="age", help="name of the age column (default age)")
    ap.add_argument("--sex-col", default="sex", help="name of the sex column (default sex)")
    args = ap.parse_args()

    src = Path(args.csv)
    # utf-8-sig strips an Excel byte-order mark; sep=None sniffs comma/tab/semicolon/pipe
    df = pd.read_csv(src, dtype=str, sep=None, engine="python", encoding="utf-8-sig")
    original_cols = list(df.columns)

    # match columns ignoring case and surrounding spaces
    norm = {c.strip().lower(): c for c in df.columns}
    wanted = {"zip5": args.zip_col, "age": args.age_col, "sex": args.sex_col}
    rename, missing = {}, []
    for canon, given in wanted.items():
        actual = norm.get(given.strip().lower())
        if actual is None:
            missing.append(given)
        else:
            rename[actual] = canon
    if missing:
        sys.exit(f"Couldn't find column(s): {missing}\n"
                 f"Columns in your file: {original_cols}\n"
                 f"Use --zip-col / --age-col / --sex-col to point at the right ones.")
    if "universe" in norm:
        sys.exit("Your file already has a 'universe' column; rename or drop it first.")
    df = df.rename(columns=rename)

    df = assign(df)
    balanced = balance_report(df)

    if args.export and not balanced and not args.force:
        sys.exit("\nNot exported because imbalance was flagged. Re-run to re-randomize, "
                 "or add --force to export this draw anyway.")
    if args.export:
        df = df.rename(columns={v: k for k, v in rename.items()})  # restore your column names
        stem = src.stem
        df.to_csv(f"{stem}_with_universe.csv", index=False)
        df[df.universe == "treatment"].drop(columns="universe").to_csv(f"{stem}_treatment.csv", index=False)
        print(f"\nWrote {stem}_with_universe.csv and {stem}_treatment.csv")


if __name__ == "__main__":
    main()
