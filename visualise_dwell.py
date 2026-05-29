import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# Load data
df = pd.read_csv('dwell_log.csv')
df['timestamp_entry'] = pd.to_datetime(df['timestamp_entry'])
df['timestamp_exit'] = pd.to_datetime(df['timestamp_exit'])

# Filter to meaningful dwell events (>= 1 second) to reduce tracker noise
df_filtered = df[df['dwell_seconds'] >= 1.0].copy()

# For the scatter plot, use timestamp_exit as the event time
df_filtered = df_filtered.sort_values('timestamp_exit')

# --- Figure setup ---
fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor('#0f1117')
gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :])   # Top: scatter over time
ax2 = fig.add_subplot(gs[1, 0])   # Bottom left: histogram
ax3 = fig.add_subplot(gs[1, 1])   # Bottom right: cumulative events

ACCENT = '#00c9a7'
TEXT   = '#e0e0e0'
GRID   = '#2a2a3a'
BG     = '#1a1a2e'

for ax in [ax1, ax2, ax3]:
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

# --- Plot 1: Dwell time scatter over time ---
scatter = ax1.scatter(
    df_filtered['timestamp_exit'],
    df_filtered['dwell_seconds'],
    c=df_filtered['dwell_seconds'],
    cmap='plasma',
    alpha=0.75,
    s=40,
    edgecolors='none'
)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=20, ha='right')
ax1.set_xlabel('Time (UTC)')
ax1.set_ylabel('Dwell Time (seconds)')
ax1.set_title('Pedestrian Dwell Time at Crossing Zone — Jackson Hole Town Square (28 May 2026)', fontsize=11, pad=10)
ax1.yaxis.grid(True, color=GRID, linestyle='--', linewidth=0.6)
ax1.set_axisbelow(True)
cb = fig.colorbar(scatter, ax=ax1, pad=0.01)
cb.set_label('Seconds', color=TEXT, fontsize=8)
cb.ax.yaxis.set_tick_params(color=TEXT)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT)

# --- Plot 2: Histogram of dwell times ---
bins = [1, 3, 5, 10, 20, 30, 45, 60, 90]
ax2.hist(df_filtered['dwell_seconds'], bins=bins, color=ACCENT, edgecolor='#0f1117', linewidth=0.5)
ax2.set_xlabel('Dwell Time (seconds)')
ax2.set_ylabel('Number of Events')
ax2.set_title('Distribution of Dwell Times (≥1s)', fontsize=10)
ax2.yaxis.grid(True, color=GRID, linestyle='--', linewidth=0.6)
ax2.set_axisbelow(True)

# --- Plot 3: Cumulative detections over time ---
df_sorted = df_filtered.sort_values('timestamp_exit').reset_index(drop=True)
df_sorted['cumulative'] = range(1, len(df_sorted) + 1)
ax3.plot(df_sorted['timestamp_exit'], df_sorted['cumulative'], color=ACCENT, linewidth=1.8)
ax3.fill_between(df_sorted['timestamp_exit'], df_sorted['cumulative'], alpha=0.15, color=ACCENT)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=20, ha='right')
ax3.set_xlabel('Time (UTC)')
ax3.set_ylabel('Cumulative Detections')
ax3.set_title('Cumulative Dwell Events Over Time', fontsize=10)
ax3.yaxis.grid(True, color=GRID, linestyle='--', linewidth=0.6)
ax3.set_axisbelow(True)

# --- Stats annotation ---
n = len(df_filtered)
mean_d = df_filtered['dwell_seconds'].mean()
median_d = df_filtered['dwell_seconds'].median()
max_d = df_filtered['dwell_seconds'].max()
stats_text = f"n={n} events ≥1s  |  mean={mean_d:.1f}s  |  median={median_d:.1f}s  |  max={max_d:.1f}s"
fig.text(0.5, 0.01, stats_text, ha='center', fontsize=9, color='#888888')

fig.suptitle('IoT Big Data Pipeline — Dwell Time Extraction from Public Livestream', 
             fontsize=13, color=TEXT, y=1.01, fontweight='bold')

plt.savefig('/mnt/user-data/outputs/dwell_visualisation.png', 
            dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved.")
print(f"Stats: {stats_text}")
