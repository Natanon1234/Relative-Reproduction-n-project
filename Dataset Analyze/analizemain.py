import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


class Target:

  def __init__(self, name):
    self.name = name
    self.count = 0

def load_dataset(folder_path, file_type, mapping):
  """Loads dataset from parquet, csv, or structured files, applies column mapping, and returns a DataFrame."""
  print(f"Loading dataset from {folder_path}...")
  try:
    if file_type == "parquet":
      try:
        # Tries to read the folder as a unified parquet dataset
        df = pd.read_parquet(folder_path, engine="pyarrow")
      except Exception:
        # Fallback: Read ALL parquet files and concatenate
        parquet_files = glob.glob(f"{folder_path}/*.parquet")
        if not parquet_files:
          return pd.DataFrame()
        
        dfs = [pd.read_parquet(f, engine="pyarrow") for f in parquet_files]
        df = pd.concat(dfs, ignore_index=True)

    elif file_type == "csv":
      dfs = []
      if os.path.isdir(folder_path):
        csv_files = glob.glob(f"{folder_path}/*.csv") + glob.glob(f"{folder_path}/*.tsv")
      else:
        csv_files = [folder_path]

      for target_file in csv_files:
        sep = "\t" if target_file.endswith(".tsv") else ","
        try:
          temp_df = pd.read_csv(target_file, sep=sep, encoding="utf-8")
        except UnicodeDecodeError:
          print(f"UTF-8 decoding failed for {target_file}. Retrying with 'latin-1' encoding...")
          temp_df = pd.read_csv(target_file, sep=sep, encoding="latin-1")
        dfs.append(temp_df)
      
      df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
      
    else:
      df = pd.DataFrame()
      
  except Exception as e:
    print(f"Error loading dataset: {e}")
    return pd.DataFrame()

  if df.empty:
    return df

  if mapping:
    df = df.rename(columns=mapping)

  return df

def preprocess_data(df, db_name, affinity_threshold):
  """Cleans data, validates required columns, forces numeric types, and filters binders."""
  if db_name == "abbind":
    df = df.rename(
        columns={
            "Partners(A_B)": "antigen_sequence",
            "Protein-1": "heavy_sequence",
            "Protein-2": "light_sequence",
            "ddG(kcal/mol)": "affinity",
        }
    )

  required_cols = [
      "heavy_sequence",
      "light_sequence",
      "antigen_sequence",
      "affinity",
  ]
  missing_cols = [col for col in required_cols if col not in df.columns]

  if missing_cols:
    rename_fallback = {}
    for col in df.columns:
      col_lower = col.lower()
      if "heavy" in col_lower or "protein-1" in col_lower:
        rename_fallback[col] = "heavy_sequence"
      elif "light" in col_lower or "protein-2" in col_lower:
        rename_fallback[col] = "light_sequence"
      elif (
          "partner" in col_lower
          or "antigen" in col_lower
          or "target" in col_lower
      ):
        rename_fallback[col] = "antigen_sequence"
      elif "affinity" in col_lower or "ddg" in col_lower or "kd" in col_lower:
        rename_fallback[col] = "affinity"

    if rename_fallback:
      df = df.rename(columns=rename_fallback)
      missing_cols = [col for col in required_cols if col not in df.columns]

  if missing_cols:
    print(f"Missing columns: {missing_cols}")
    return pd.DataFrame(), pd.DataFrame()

  df["affinity"] = pd.to_numeric(df["affinity"], errors="coerce")
  df = df.dropna(
      subset=["heavy_sequence", "light_sequence", "antigen_sequence", "affinity"]
  )

  if df.empty:
    return pd.DataFrame(), pd.DataFrame()

  if "affinity_type" in df.columns:
    bindto_df = df.copy()
  else:
    bindto_df = df.copy()

  return df, bindto_df


def generate_controls(df):
  """Generates synthetic Pareto distribution samples and true random sample pairings."""
  print("Generating expanded synthetic distribution samples...")
  np.random.seed(42)
  synthetic_size = 50000
  synthetic_ab_counts = (
      np.random.pareto(a=1.5, size=synthetic_size).astype(int) + 1
  )
  synthetic_ag_counts = (
      np.random.pareto(a=1.8, size=synthetic_size).astype(int) + 1
  )

  print("Generating true random sample pairing...")
  unique_antigens = df["antigen_sequence"].dropna().unique()
  unique_heavy = df["heavy_sequence"].dropna().unique()
  unique_light = df["light_sequence"].dropna().unique()

  random_sample_size = min(len(df), 20000)
  random_antigens = (
      np.random.choice(unique_antigens, size=random_sample_size)
      if len(unique_antigens) > 0
      else []
  )
  random_heavy = (
      np.random.choice(unique_heavy, size=random_sample_size)
      if len(unique_heavy) > 0
      else []
  )
  random_light = (
      np.random.choice(unique_light, size=random_sample_size)
      if len(unique_light) > 0
      else []
  )

  random_df = pd.DataFrame({
      "antigen_sequence": random_antigens,
      "heavy_sequence": random_heavy,
      "light_sequence": random_light,
  })

  if not random_df.empty:
    random_df["antibody_id"] = (
        random_df["heavy_sequence"] + "_" + random_df["light_sequence"]
    )
  else:
    random_df["antibody_id"] = []

  return synthetic_ab_counts, synthetic_ag_counts, random_df


def compute_frequencies(
    bindto_df, random_df, synthetic_ab_counts, synthetic_ag_counts
):
  """Computes occurrence frequency distributions for real, synthetic, and random sets."""
  if not bindto_df.empty:
    ab_per_ag_bind = (
        bindto_df.groupby("antigen_sequence")
        .size()
        .reset_index(name="num_ab")
    )
    ab_freq_bind = ab_per_ag_bind["num_ab"].value_counts().sort_index()

    bindto_df["antibody_id"] = (
        bindto_df["heavy_sequence"] + "_" + bindto_df["light_sequence"]
    )
    ag_per_ab_bind = (
        bindto_df.groupby("antibody_id").size().reset_index(name="num_ag")
    )
    ag_freq_bind = ag_per_ab_bind["num_ag"].value_counts().sort_index()
  else:
    ab_freq_bind = pd.Series(dtype=int)
    ag_freq_bind = pd.Series(dtype=int)

  synth_ab_freq = pd.Series(synthetic_ab_counts).value_counts().sort_index()
  synth_ag_freq = pd.Series(synthetic_ag_counts).value_counts().sort_index()

  if not random_df.empty:
    ab_per_ag_rand = (
        random_df.groupby("antigen_sequence").size().reset_index(name="num_ab")
    )
    ab_freq_rand = ab_per_ag_rand["num_ab"].value_counts().sort_index()

    ag_per_ab_rand = (
        random_df.groupby("antibody_id").size().reset_index(name="num_ag")
    )
    ag_freq_rand = ag_per_ab_rand["num_ag"].value_counts().sort_index()
  else:
    ab_freq_rand = pd.Series(dtype=int)
    ag_freq_rand = pd.Series(dtype=int)

  return (
      ab_freq_bind,
      ag_freq_bind,
      synth_ab_freq,
      synth_ag_freq,
      ab_freq_rand,
      ag_freq_rand,
  )


def plot_and_save_charts(db_name, freqs, output_dir):
  """Generates and autosaves both 2x2 comparison line and bar charts for a given dataset."""
  (
      ab_freq_bind,
      ag_freq_bind,
      synth_ab_freq,
      synth_ag_freq,
      ab_freq_rand,
      ag_freq_rand,
  ) = freqs

  plt.style.use(
      "seaborn-v0_8-whitegrid"
      if "seaborn-v0_8-whitegrid" in plt.style.available
      else "default"
  )

  # --- 1. Bar Chart Generation ---
  print(
      f"Generating and saving bar charts for [{db_name}]..."
  )
  fig, axes = plt.subplots(2, 2, figsize=(16, 12))
  width = 0.6

  # Row 1: Antibodies per Antigen (Bar)
  axes[0, 0].bar(
      ab_freq_bind.index,
      ab_freq_bind.values,
      width=width,
      color="#1f77b4",
      alpha=0.8,
      label=f"{db_name} Binders",
  )
  axes[0, 0].bar(
      synth_ab_freq.index,
      synth_ab_freq.values,
      width=width * 0.7,
      color="#2ca02c",
      alpha=0.5,
      label="Synthetic",
  )
#   axes[0, 0].bar( ab_freq_rand.index,
#       ab_freq_rand.values,
#       width=width * 0.4,
#       color="#9467bd",
#       alpha=0.5,
#       label="True Random",
#   )
  axes[0, 0].set_title(
      f"[{db_name}] Ab per Antigen (Linear Bar Chart)",
      fontsize=11,
      fontweight="bold",
  )
  axes[0, 0].set_xlabel("Number of Associated Antibodies (Ab)")
  axes[0, 0].set_ylabel("Count")
  axes[0, 0].legend()
  axes[0, 0].grid(True, linestyle="--", alpha=0.5)

  axes[0, 1].bar(
      ab_freq_bind.index,
      ab_freq_bind.values,
      width=width,
      color="#1f77b4",
      alpha=0.8,
      label=f"{db_name} Binders",
  )
#   axes[0, 1].bar(
#       synth_ab_freq.index,
#       synth_ab_freq.values,
#       width=width * 0.7,
#       color="#2ca02c",
#       alpha=0.5,
#       label="Synthetic",
#   )
#   axes[0, 1].bar(
#       ab_freq_rand.index,
#       ab_freq_rand.values,
#       width=width * 0.4,
#       color="#9467bd",
#       alpha=0.5,
#       label="True Random",
#   )
  axes[0, 1].set_title(
      f"[{db_name}] Ab per Antigen (Log Scale Bar Chart)",
      fontsize=11,
      fontweight="bold",
  )
  axes[0, 1].set_xlabel("Number of Associated Antibodies (Ab)")
  axes[0, 1].set_ylabel("Count (Log Scale)")
  axes[0, 1].set_yscale("log")
  axes[0, 1].set_xscale("log")
  axes[0, 1].legend()
  axes[0, 1].grid(True, which="both", linestyle="--", alpha=0.5)

  # Row 2: Antigens per Antibody (Bar)
  axes[1, 0].bar(
      ag_freq_bind.index,
      ag_freq_bind.values,
      width=width,
      color="#ff7f0e",
      alpha=0.8,
      label=f"{db_name} Binders",
  )
  axes[1, 0].bar(
      synth_ag_freq.index,
      synth_ag_freq.values,
      width=width * 0.7,
      color="#d62728",
      alpha=0.5,
      label="Synthetic",
  )
  axes[1, 0].bar(
      ag_freq_rand.index,
      ag_freq_rand.values,
      width=width * 0.4,
      color="#8c564b",
      alpha=0.5,
      label="True Random",
  )
  axes[1, 0].set_title(
      f"[{db_name}] Ag per Antibody (Linear Bar Chart)",
      fontsize=11,
      fontweight="bold",
  )
  axes[1, 0].set_xlabel("Number of Associated Antigens (Ag)")
  axes[1, 0].set_ylabel("Count")
  axes[1, 0].legend()
  axes[1, 0].grid(True, linestyle="--", alpha=0.5)

  axes[1, 1].bar(
      ag_freq_bind.index,
      ag_freq_bind.values,
      width=width,
      color="#ff7f0e",
      alpha=0.8,
      label=f"{db_name} Binders",
  )
  axes[1, 1].bar(
      synth_ag_freq.index,
      synth_ag_freq.values,
      width=width * 0.7,
      color="#d62728",
      alpha=0.5,
      label="Synthetic",
  )
  axes[1, 1].bar(
      ag_freq_rand.index,
      ag_freq_rand.values,
      width=width * 0.4,
      color="#8c564b",
      alpha=0.5,
      label="True Random",
  )
  axes[1, 1].set_title(
      f"[{db_name}] Ag per Antibody (Log Scale Bar Chart)",
      fontsize=11,
      fontweight="bold",
  )
  axes[1, 1].set_xlabel("Number of Associated Antigens (Ag)")
  axes[1, 1].set_ylabel("Count (Log Scale)")
  axes[1, 1].set_yscale("log")
  axes[1, 1].set_xscale("log")
  axes[1, 1].legend()
  axes[1, 1].grid(True, which="both", linestyle="--", alpha=0.5)

  plt.tight_layout()
  bar_figure_path = os.path.join(
      output_dir, f"{db_name}_ab_ag_distribution_comparison_bar.png"
  )
  plt.savefig(bar_figure_path, dpi=300)
  print(f"Bar chart for [{db_name}] successfully saved to: {bar_figure_path}")
  plt.close()

  # --- 2. Line Chart Generation ---
  print(
      f"Generating and saving comparative line charts for [{db_name}]..."
  )
  fig, axes = plt.subplots(2, 2, figsize=(16, 12))

  # Row 1: Antibodies per Antigen (Line)
  axes[0, 0].plot(
      ab_freq_bind.index,
      ab_freq_bind.values,
      marker="o",
      linewidth=2,
      color="#1f77b4",
      label=f"{db_name} Binders",
  )
  axes[0, 0].plot(
      synth_ab_freq.index,
      synth_ab_freq.values,
      marker="s",
      linewidth=2,
      color="#2ca02c",
      label="Synthetic",
  )
  axes[0, 0].plot(
      ab_freq_rand.index,
      ab_freq_rand.values,
      marker="^",
      linewidth=2,
      color="#9467bd",
      label="True Random",
  )
  axes[0, 0].set_title(
      f"[{db_name}] Ab per Antigen (Linear Line Chart)",
      fontsize=11,
      fontweight="bold",
  )
  axes[0, 0].set_xlabel("Number of Associated Antibodies (Ab)")
  axes[0, 0].set_ylabel("Count")
  axes[0, 0].legend()
  axes[0, 0].grid(True, linestyle="--", alpha=0.5)

  axes[0, 1].plot(
      ab_freq_bind.index,
      ab_freq_bind.values,
      marker="o",
      linewidth=2,
      color="#1f77b4",
      label=f"{db_name} Binders",
  )
  axes[0, 1].plot(
      synth_ab_freq.index,
      synth_ab_freq.values,
      marker="s",
      linewidth=2,
      color="#2ca02c",
      label="Synthetic",
  )
  axes[0, 1].plot(
      ab_freq_rand.index,
      ab_freq_rand.values,
      marker="^",
      linewidth=2,
      color="#9467bd",
      label="True Random",
  )
  axes[0, 1].set_title(
      f"[{db_name}] Ab per Antigen (Log Scale Line Chart)",
      fontsize=11,
      fontweight="bold",
  )
  axes[0, 1].set_xlabel("Number of Associated Antibodies (Ab)")
  axes[0, 1].set_ylabel("Count (Log Scale)")
  axes[0, 1].set_yscale("log")
  axes[0, 1].set_xscale("log")
  axes[0, 1].legend()
  axes[0, 1].grid(True, which="both", linestyle="--", alpha=0.5)

  # Row 2: Antigens per Antibody (Line)
  axes[1, 0].plot(
      ag_freq_bind.index,
      ag_freq_bind.values,
      marker="o",
      linewidth=2,
      color="#ff7f0e",
      label=f"{db_name} Binders",
  )
  axes[1, 0].plot(
      synth_ag_freq.index,
      synth_ag_freq.values,
      marker="s",
      linewidth=2,
      color="#d62728",
      label="Synthetic",
  )
  axes[1, 0].plot(
      ag_freq_rand.index,
      ag_freq_rand.values,
      marker="^",
      linewidth=2,
      color="#8c564b",
      label="True Random",
  )
  axes[1, 0].set_title(
      f"[{db_name}] Ag per Antibody (Linear Line Chart)",
      fontsize=11,
      fontweight="bold",
  )
  axes[1, 0].set_xlabel("Number of Associated Antigens (Ag)")
  axes[1, 0].set_ylabel("Count")
  axes[1, 0].legend()
  axes[1, 0].grid(True, linestyle="--", alpha=0.5)

  axes[1, 1].plot(
      ag_freq_bind.index,
      ag_freq_bind.values,
      marker="o",
      linewidth=2,
      color="#ff7f0e",
      label=f"{db_name} Binders",
  )
  axes[1, 1].plot(
      synth_ag_freq.index,
      synth_ag_freq.values,
      marker="s",
      linewidth=2,
      color="#d62728",
      label="Synthetic",
  )
  axes[1, 1].plot(
      ag_freq_rand.index,
      ag_freq_rand.values,
      marker="^",
      linewidth=2,
      color="#8c564b",
      label="True Random",
  )
  axes[1, 1].set_title(
      f"[{db_name}] Ag per Antibody (Log Scale Line Chart)",
      fontsize=11,
      fontweight="bold",
  )
  axes[1, 1].set_xlabel("Number of Associated Antigens (Ag)")
  axes[1, 1].set_ylabel("Count (Log Scale)")
  axes[1, 1].set_yscale("log")
  axes[1, 1].set_xscale("log")
  axes[1, 1].legend()
  axes[1, 1].grid(True, which="both", linestyle="--", alpha=0.5)

  plt.tight_layout()
  line_figure_path = os.path.join(
      output_dir, f"{db_name}_ab_ag_distribution_comparison_line.png"
  )
  plt.savefig(line_figure_path, dpi=300)
  print(f"Line chart for [{db_name}] successfully saved to: {line_figure_path}")
  plt.close()


if __name__ == "__main__":
  affinity_threshold = 10.0
  output_dir = "/Users/natanon/Documents/CICM/Hokkaido University/Dataset Analyze/output_results"
  os.makedirs(output_dir, exist_ok=True)

  datasets_config = {
      "asd": {
          "path": "/Users/natanon/Documents/CICM/Hokkaido University/Dataset Analyze/datasets/asd",
          "type": "parquet",
          "mapping": {},
      },
      "abbind": {
          "path": "/Users/natanon/Documents/CICM/Hokkaido University/Dataset Analyze/datasets/abbind",
          "type": "csv",
          "mapping": {},
      },
  }

  for db_name, config in datasets_config.items():
    print(f"\n==========================================")
    print(f"Processing Database: {db_name}")
    print(f"==========================================")

    df = load_dataset(config["path"], config["type"], config["mapping"])
    if db_name == "asd":
        print(f"ASD columns loaded: {df.columns.tolist()}")
        print(f"ASD row count before filtering: {len(df)}")
    if df.empty:
      print(f"Skipping {db_name}: Dataset path yielded no records.")
      continue

    df, bindto_df = preprocess_data(df, db_name, affinity_threshold)
    if df.empty:
      print(f"Skipping {db_name}: No valid rows remaining after filtering.")
      continue

    pair_counts = (
        df.groupby(["antigen_sequence", "heavy_sequence", "light_sequence"])
        .size()
        .reset_index(name="count")
    )
    targets = [
        Target(row.antigen_sequence)
        for row in tqdm(
            pair_counts.itertuples(),
            total=len(pair_counts),
            desc=f"Building targets ({db_name})",
        )
    ]

    synth_ab, synth_ag, random_df = generate_controls(df)
    freqs = compute_frequencies(bindto_df, random_df, synth_ab, synth_ag)
    plot_and_save_charts(db_name, freqs, output_dir)
    print(df)
    print(bindto_df)