import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 1. Load and clean dataset
# =========================
file_path = Path('/mnt/data/newotp3months - Sheet1.csv')
df = pd.read_csv(file_path)

# Normalize column names
df.columns = [c.strip().lower().replace('-', '_').replace(' ', '_') for c in df.columns]

# Map to expected names
col_map = {}
for target in ['date', 'otp_count', 'user_count']:
    match = [c for c in df.columns if c == target]
    if not match:
        match = [c for c in df.columns if target.replace('_','') in c.replace('_','')]
    if match:
        col_map[target] = match[0]

df = df.rename(columns={v: k for k, v in col_map.items()})

# Keep relevant columns
df = df[['date','otp_count','user_count']].copy()

# Parse types
df['date'] = pd.to_datetime(df['date'], errors='coerce', infer_datetime_format=True)
df['otp_count'] = pd.to_numeric(df['otp_count'], errors='coerce')
df['user_count'] = pd.to_numeric(df['user_count'], errors='coerce')

# Drop bad rows
df = df.dropna().astype({'otp_count':'int','user_count':'int'})

# Aggregate in case duplicates exist
df = df.groupby(['date','otp_count'], as_index=False)['user_count'].sum().sort_values(['date','otp_count'])

# =========================================
# 2. Daily summary: distribution & metrics
# =========================================
daily = df.pivot_table(index='date', columns='otp_count', values='user_count', aggfunc='sum', fill_value=0)
daily.columns.name = 'otp_count'

numeric_counts = [c for c in daily.columns if isinstance(c, (int, np.integer))]
daily['total_users'] = daily[numeric_counts].sum(axis=1)

# Shares at various thresholds
for k in [3, 5, 10, 16, 17, 20, 30]:
    ge_cols = [c for c in numeric_counts if c >= k]
    daily[f'share_ge_{k}'] = daily[ge_cols].sum(axis=1) / daily['total_users']

# Weighted mean OTPs per user/day
daily['weighted_mean_otps'] = (
    daily[numeric_counts]
    .mul(pd.Series(numeric_counts, index=numeric_counts), axis=1)
    .sum(axis=1) / daily['total_users']
)

# Flatten summary
daily_summary = daily[['total_users','weighted_mean_otps',
                       'share_ge_3','share_ge_5','share_ge_10',
                       'share_ge_16','share_ge_17','share_ge_20','share_ge_30']].reset_index()

# Save to CSV
report_path = Path('/mnt/data/otp_daily_summary_report.csv')
daily_summary.to_csv(report_path, index=False)
print(f"Report saved at: {report_path}")

# ===================================
# 3. Overall statistics (decision-use)
# ===================================
total_users = df['user_count'].sum()
share1 = df.loc[df['otp_count']==1,'user_count'].sum() / total_users
share2 = df.loc[df['otp_count']==2,'user_count'].sum() / total_users
share_ge3 = df.loc[df['otp_count']>=3,'user_count'].sum() / total_users
share_ge16 = df.loc[df['otp_count']>=16,'user_count'].sum() / total_users

print("Overall distribution (user-weighted across all days):")
print(f"  One OTP:        {share1:.2%}")
print(f"  Two OTPs:       {share2:.2%}")
print(f"  ≥3 OTPs:        {share_ge3:.2%}")
print(f"  ≥16 OTPs/day:   {share_ge16:.2%}")
print(f"  Weighted mean OTPs/user/day: {daily['weighted_mean_otps'].mean():.2f}")

# ======================
# 4. Plots for insights
# ======================

# Distribution across all days
dist = df.groupby('otp_count')['user_count'].sum().sort_index()
plt.figure(figsize=(8,5))
plt.plot(dist.index, dist.values, marker='o')
plt.title('Distribution of OTP counts per user per day')
plt.xlabel('OTP count')
plt.ylabel('Total users across days')
plt.yscale('log')  # log scale to see long tail
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Time series of heavy OTP requesters (>=17/day)
plt.figure(figsize=(10,5))
plt.plot(daily_summary['date'], daily_summary['share_ge_17'])
plt.title('Share of users requesting >=17 OTPs in a day')
plt.xlabel('Date')
plt.ylabel('Share of users')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Weighted mean OTPs per user/day over time
plt.figure(figsize=(10,5))
plt.plot(daily_summary['date'], daily_summary['weighted_mean_otps'])
plt.title('Weighted mean OTPs per user per day')
plt.xlabel('Date')
plt.ylabel('Mean OTPs')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
