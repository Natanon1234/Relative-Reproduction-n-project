# Pipeline note: Identify antigens --> Tupple of variant, count --> Plotting (Ab per Ag / Ag per Ab) as Bar Charts
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set up the functions
affinity = 10.0 # cut point for binding affinity (e.g., Kd < 10)

def occurances(name, list):
    count=0
    for i in list:
        if name == i:
            count=count+1
    return count

# Execute the program

# Set up class
class target:
    def __init__(self, name):
        self.name=name
        self.count=0

# Setup Autosave Directory for figures
output_dir = 'output_results'
os.makedirs(output_dir, exist_ok=True)

# File parsing: Loading the partitioned Parquet files from your specified directory
folder_path = '/Users/natanon/Documents/CICM/Hokkaido University/Dataset Analyze/datasets/asd'

print("Loading dataset...")
try:
    df = pd.read_parquet(folder_path, engine='pyarrow')
except Exception:
    parquet_files = glob.glob(f'{folder_path}/*.parquet')
    df = pd.read_parquet(parquet_files, engine='pyarrow')

# Force the 'affinity' column to be numeric numbers (floats) 
df['affinity'] = pd.to_numeric(df['affinity'], errors='coerce')

# Clean out empty rows so our code doesn't break
df = df.dropna(subset=['heavy_sequence', 'light_sequence', 'antigen_sequence', 'affinity'])

# Identify all Ag / Ab with a progress indicator for the loop
targets=[]
pair_counts = df.groupby(['antigen_sequence', 'heavy_sequence', 'light_sequence']).size().reset_index(name='count')

print("Processing unique targets...")
for row in tqdm(pair_counts.itertuples(), total=len(pair_counts), desc="Building target objects"):
    unique_target = target(name=row.antigen_sequence)
    unique_target.count = row.count
    targets.append(unique_target)

# 1. Filter for real binders (affinity < threshold)
bindto_df = df[(df['affinity_type'] == 'kd') & (df['affinity'] < affinity)].copy()

# 2. Simulate expanded synthetic pairing (Pareto distribution)
print("Generating expanded synthetic distribution samples...")
np.random.seed(42)
synthetic_size = 50000
synthetic_ab_counts = np.random.pareto(a=1.5, size=synthetic_size).astype(int) + 1
synthetic_ag_counts = np.random.pareto(a=1.8, size=synthetic_size).astype(int) + 1

# 3. True Random Sample Pairing
print("Generating true random sample pairing...")
unique_antigens = df['antigen_sequence'].dropna().unique()
unique_heavy = df['heavy_sequence'].dropna().unique()
unique_light = df['light_sequence'].dropna().unique()

random_sample_size = min(len(df), 20000)
random_antigens = np.random.choice(unique_antigens, size=random_sample_size)
random_heavy = np.random.choice(unique_heavy, size=random_sample_size)
random_light = np.random.choice(unique_light, size=random_sample_size)

random_df = pd.DataFrame({
    'antigen_sequence': random_antigens,
    'heavy_sequence': random_heavy,
    'light_sequence': random_light
})
random_df['antibody_id'] = random_df['heavy_sequence'] + "_" + random_df['light_sequence']


# ---------------------------------------------------------
# COMPUTING FREQUENCIES FOR PLOTTING
# ---------------------------------------------------------

# Real Binders Metrics
ab_per_ag_bind = bindto_df.groupby('antigen_sequence').size().reset_index(name='num_ab')
ab_freq_bind = ab_per_ag_bind['num_ab'].value_counts().sort_index()

bindto_df['antibody_id'] = bindto_df['heavy_sequence'] + "_" + bindto_df['light_sequence']
ag_per_ab_bind = bindto_df.groupby('antibody_id').size().reset_index(name='num_ag')
ag_freq_bind = ag_per_ab_bind['num_ag'].value_counts().sort_index()

# Synthetic Distribution Frequency Mapping
synth_ab_freq = pd.Series(synthetic_ab_counts).value_counts().sort_index()
synth_ag_freq = pd.Series(synthetic_ag_counts).value_counts().sort_index()

# True Random Sample Frequency Mapping
ab_per_ag_rand = random_df.groupby('antigen_sequence').size().reset_index(name='num_ab')
ab_freq_rand = ab_per_ag_rand['num_ab'].value_counts().sort_index()

ag_per_ab_rand = random_df.groupby('antibody_id').size().reset_index(name='num_ag')
ag_freq_rand = ag_per_ab_rand['num_ag'].value_counts().sort_index()


# ---------------------------------------------------------
# PLOTTING SECTION (BAR CHARTS) & AUTOSAVE GRAPH ONLY
# ---------------------------------------------------------

print("Generating and saving comparative bar charts...")

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Width adjustments for overlapping bar charts visibility
width = 0.6

# --- Row 1: Antibodies per Antigen (Linear vs Log Bar Charts) ---
# Linear Scale
axes[0, 0].bar(ab_freq_bind.index, ab_freq_bind.values, width=width, color='#1f77b4', alpha=0.8, label='Real Binders')
axes[0, 0].bar(synth_ab_freq.index, synth_ab_freq.values, width=width*0.7, color='#2ca02c', alpha=0.5, label='Synthetic')
axes[0, 0].bar(ab_freq_rand.index, ab_freq_rand.values, width=width*0.4, color='#9467bd', alpha=0.5, label='True Random')
axes[0, 0].set_title('Ab per Antigen (Linear Bar Chart)', fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel('Number of Associated Antibodies (Ab)')
axes[0, 0].set_ylabel('Count')
axes[0, 0].legend()
axes[0, 0].grid(True, linestyle='--', alpha=0.5)

# Log Scale
axes[0, 1].bar(ab_freq_bind.index, ab_freq_bind.values, width=width, color='#1f77b4', alpha=0.8, label='Real Binders')
axes[0, 1].bar(synth_ab_freq.index, synth_ab_freq.values, width=width*0.7, color='#2ca02c', alpha=0.5, label='Synthetic')
axes[0, 1].bar(ab_freq_rand.index, ab_freq_rand.values, width=width*0.4, color='#9467bd', alpha=0.5, label='True Random')
axes[0, 1].set_title('Ab per Antigen (Log Scale Bar Chart)', fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('Number of Associated Antibodies (Ab)')
axes[0, 1].set_ylabel('Count (Log Scale)')
axes[0, 1].set_yscale('log')
axes[0, 1].set_xscale('log')
axes[0, 1].legend()
axes[0, 1].grid(True, which="both", linestyle='--', alpha=0.5)


# --- Row 2: Antigens per Antibody (Linear vs Log Bar Charts) ---
# Linear Scale
axes[1, 0].bar(ag_freq_bind.index, ag_freq_bind.values, width=width, color='#ff7f0e', alpha=0.8, label='Real Binders')
axes[1, 0].bar(synth_ag_freq.index, synth_ag_freq.values, width=width*0.7, color='#d62728', alpha=0.5, label='Synthetic')
axes[1, 0].bar(ag_freq_rand.index, ag_freq_rand.values, width=width*0.4, color='#8c564b', alpha=0.5, label='True Random')
axes[1, 0].set_title('Ag per Antibody (Linear Bar Chart)', fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel('Number of Associated Antigens (Ag)')
axes[1, 0].set_ylabel('Count')
axes[1, 0].legend()
axes[1, 0].grid(True, linestyle='--', alpha=0.5)

# Log Scale
axes[1, 1].bar(ag_freq_bind.index, ag_freq_bind.values, width=width, color='#ff7f0e', alpha=0.8, label='Real Binders')
axes[1, 1].bar(synth_ag_freq.index, synth_ag_freq.values, width=width*0.7, color='#d62728', alpha=0.5, label='Synthetic')
axes[1, 1].bar(ag_freq_rand.index, ag_freq_rand.values, width=width*0.4, color='#8c564b', alpha=0.5, label='True Random')
axes[1, 1].set_title('Ag per Antibody (Log Scale Bar Chart)', fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel('Number of Associated Antigens (Ag)')
axes[1, 1].set_ylabel('Count (Log Scale)')
axes[1, 1].set_yscale('log')
axes[1, 1].set_xscale('log')
axes[1, 1].legend()
axes[1, 1].grid(True, which="both", linestyle='--', alpha=0.5)

plt.tight_layout()

# Autosave Figure Only
figure_path = os.path.join(output_dir, 'ab_ag_distribution_comparison_bar.png')
plt.savefig(figure_path, dpi=300)
print(f"Bar chart successfully saved to: {figure_path}")

plt.show()