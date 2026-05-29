"""Experiment 3: Residual friction analysis."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.metrics.residual_friction import compute_rif
from src.metrics.expected_cost_model import coefficients, fit, _check_vif
from sklearn.preprocessing import StandardScaler

RES_DIR = ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)
BASE_DIR = ROOT / "results/baseline_rankings"
BASE_DIR.mkdir(parents=True, exist_ok=True)


def run(size: str = "small", tokenizer: str = "custom_bpe", model_type: str = "poisson"):
    df = pd.read_parquet(ROOT / "data/processed" / f"word_metrics_{size}.parquet")
    df = df[df["tokenizer"] == tokenizer].copy()

    df, pipe, feat_cols = compute_rif(df, model_type=model_type)

    # Save VIF table alongside coefficients
    from src.metrics.expected_cost_model import _add_char_features  # noqa: PLC0415
    df_feat = _add_char_features(df)
    X_scaled = StandardScaler().fit_transform(df_feat[feat_cols].fillna(0).values)
    vif_df = _check_vif(X_scaled, feat_cols)
    vif_df.to_csv(RES_DIR / "feature_vif.csv", index=False)
    print(f"\nVIF:\n{vif_df.to_string(index=False)}")

    coefs = coefficients(pipe, feat_cols)
    coefs.to_csv(RES_DIR / "residual_model_coefficients.csv", index=False)

    top_cols = ["word", "freq", "doc_freq", "tfidf_top_k_mean", "token_count",
                "expected_cost", "positive_residual", "RIF"]
    df.nlargest(200, "RIF")[top_cols].to_csv(RES_DIR / "top_rif_words.csv", index=False)

    # Baselines
    df.nlargest(200, "tfidf_top_k_mean")[top_cols].to_csv(BASE_DIR / "top_tfidf.csv", index=False)
    df.nlargest(200, "token_count")[top_cols].to_csv(BASE_DIR / "top_token_count.csv", index=False)
    df.nlargest(200, "tokens_per_char")[top_cols].to_csv(BASE_DIR / "top_tokens_per_char.csv", index=False)
    df.nsmallest(200, "freq")[top_cols].to_csv(BASE_DIR / "top_rare.csv", index=False)
    df.nlargest(200, "freq")[top_cols].to_csv(BASE_DIR / "top_frequent.csv", index=False)

    print("Residual model coefficients:")
    print(coefs.to_string(index=False))
    print(f"\nTop 20 RIF words:\n{df.nlargest(20, 'RIF')['word'].tolist()}")
    print(f"\nResidual cost stats:\n{df['residual_cost'].describe()}")

    return df


if __name__ == "__main__":
    import argparse, yaml  # noqa: E401

    p = argparse.ArgumentParser()
    p.add_argument("--size", default="small")
    p.add_argument("--tokenizer", default="custom_bpe")
    args = p.parse_args()

    with open(ROOT / "configs/experiments.yaml") as f:
        exp_cfg = yaml.safe_load(f)

    run(args.size, args.tokenizer, exp_cfg["residual_model"])
