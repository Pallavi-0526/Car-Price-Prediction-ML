# ============================================================
#   CAR PRICE PREDICTION
# ============================================================

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("Loading model data...")
with open('model_data.pkl', 'rb') as f:
    d = pickle.load(f)

df     = d['df'];    df2   = d['df2']
X_test = d['X_test']; y_test = d['y_test']
lr_p   = d['lr_p'];   rf_p  = d['rf_p']
lr_r2  = d['lr_r2'];  rf_r2 = d['rf_r2']
lr_mae = d['lr_mae']; rf_mae= d['rf_mae']
lr_rmse= d['lr_rmse'];rf_rmse=d['rf_rmse']
fi_df  = d['fi_df']

BG   = '#0d0d14'
S1   = '#16161f'
ACC  = '#e8ff47'
ACC2 = '#47ffb8'
ACC3 = '#ff6b6b'
ACC4 = '#a78bfa'
TXT  = '#f0f0f5'
TXT2 = '#9898a8'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': S1,
    'axes.edgecolor': '#2a2a3a', 'axes.labelcolor': TXT2,
    'xtick.color': TXT2, 'ytick.color': TXT2,
    'text.color': TXT, 'grid.color': '#1e1e2a',
    'grid.linewidth': 0.5,
    'axes.spines.top': False, 'axes.spines.right': False,
})

fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
fig1.patch.set_facecolor(BG)
fig1.suptitle('Car Price Dataset — Exploratory Data Analysis',
              fontsize=18, fontweight='bold', color=TXT, y=0.98)

ax = axes[0, 0]
prices = df2['selling_price']
ax.hist(prices[prices < 30], bins=40, color=ACC,
        edgecolor=BG, linewidth=0.5, alpha=0.85)
ax.axvline(prices.median(), color=ACC2, linewidth=1.5,
           linestyle='--', label=f'Median ₹{prices.median():.1f}L')
ax.set_title('Selling Price Distribution', fontsize=13, color=TXT, pad=10)
ax.set_xlabel('Price (₹ Lakhs)')
ax.set_ylabel('Count')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

ax = axes[0, 1]
brand_avg = df2.groupby('brand')['selling_price'].median().sort_values()
colors = [ACC3 if v > 15 else ACC if v > 8 else ACC2 for v in brand_avg.values]
bars = ax.barh(brand_avg.index, brand_avg.values, color=colors, edgecolor=BG)
ax.set_title('Median Price by Brand', fontsize=13, color=TXT, pad=10)
ax.set_xlabel('Median Price (₹ Lakhs)')
for bar, val in zip(bars, brand_avg.values):
    ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
            f'₹{val:.1f}L', va='center', fontsize=9, color=TXT2)
ax.grid(axis='x', alpha=0.3)

ax = axes[1, 0]
sample = df2.sample(300, random_state=1)
sc = ax.scatter(sample['car_age'], sample['selling_price'],
                c=sample['selling_price'], cmap='YlGn',
                alpha=0.6, s=20, edgecolors='none')
ax.set_title('Car Age vs Selling Price', fontsize=13, color=TXT, pad=10)
ax.set_xlabel('Car Age (Years)')
ax.set_ylabel('Price (₹ Lakhs)')
ax.grid(alpha=0.2)
cbar = fig1.colorbar(sc, ax=ax, fraction=0.03)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TXT2)

ax = axes[1, 1]
fuel_counts = df2['fuel'].value_counts()
wedge_colors = [ACC, ACC2, ACC3, ACC4]
wedges, texts, autotexts = ax.pie(
    fuel_counts.values, labels=fuel_counts.index,
    autopct='%1.1f%%', colors=wedge_colors[:len(fuel_counts)],
    startangle=90, pctdistance=0.75,
    wedgeprops={'edgecolor': BG, 'linewidth': 2})
for t in texts:
    t.set_color(TXT2); t.set_fontsize(11)
for at in autotexts:
    at.set_color(BG); at.set_fontsize(10); at.set_fontweight('bold')
ax.set_title('Fuel Type Distribution', fontsize=13, color=TXT, pad=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig1.savefig('plot1_eda.png', dpi=150, bbox_inches='tight', facecolor=BG)
print(" plot1_eda.png saved!")

fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.patch.set_facecolor(BG)
fig2.suptitle('Model Evaluation — Linear Regression vs Random Forest',
              fontsize=18, fontweight='bold', color=TXT, y=0.98)

lim = max(float(y_test.max()), float(lr_p.max()), float(rf_p.max()))

ax = axes[0, 0]
ax.scatter(y_test, lr_p, alpha=0.35, s=15, color=ACC2, edgecolors='none')
ax.plot([0, lim], [0, lim], color=ACC, linewidth=1.5,
        linestyle='--', label='Perfect prediction')
ax.set_title(f'Linear Regression  (R²={lr_r2:.3f})', fontsize=13, color=TXT, pad=10)
ax.set_xlabel('Actual Price (₹L)')
ax.set_ylabel('Predicted Price (₹L)')
ax.legend(fontsize=9); ax.grid(alpha=0.2)

ax = axes[0, 1]
ax.scatter(y_test, rf_p, alpha=0.35, s=15, color=ACC, edgecolors='none')
ax.plot([0, lim], [0, lim], color=ACC2, linewidth=1.5,
        linestyle='--', label='Perfect prediction')
ax.set_title(f'Random Forest  (R²={rf_r2:.3f})', fontsize=13, color=TXT, pad=10)
ax.set_xlabel('Actual Price (₹L)')
ax.set_ylabel('Predicted Price (₹L)')
ax.legend(fontsize=9); ax.grid(alpha=0.2)

ax = axes[1, 0]
metrics = ['MAE (₹L)', 'RMSE (₹L)', 'R² Score']
lr_vals = [lr_mae, lr_rmse, lr_r2]
rf_vals = [rf_mae, rf_rmse, rf_r2]
x = np.arange(3); w = 0.35
b1 = ax.bar(x - w/2, lr_vals, w, color=ACC2, label='Linear Regression', edgecolor=BG)
b2 = ax.bar(x + w/2, rf_vals, w, color=ACC,  label='Random Forest',     edgecolor=BG)
ax.set_title('Metrics Comparison', fontsize=13, color=TXT, pad=10)
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.2f}', ha='center', fontsize=9, color=TXT2)

ax = axes[1, 1]
residuals = np.array(y_test) - rf_p
ax.hist(residuals, bins=40, color=ACC4, edgecolor=BG, linewidth=0.5, alpha=0.85)
ax.axvline(0, color=ACC, linewidth=1.5, linestyle='--', label='Zero error')
ax.set_title('RF Residuals Distribution', fontsize=13, color=TXT, pad=10)
ax.set_xlabel('Residual (Actual − Predicted)')
ax.set_ylabel('Count')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig2.savefig('plot2_model.png', dpi=150, bbox_inches='tight', facecolor=BG)
print(" plot2_model.png saved!")

fig3, ax = plt.subplots(figsize=(10, 6))
fig3.patch.set_facecolor(BG)
ax.set_facecolor(S1)

colors_fi = [ACC if i < 2 else ACC2 if i < 5 else TXT2
             for i in range(len(fi_df))]
bars = ax.barh(fi_df['Feature'], fi_df['Importance'] * 100,
               color=colors_fi, edgecolor=BG, linewidth=0.5)
for bar, val in zip(bars, fi_df['Importance'] * 100):
    ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}%', va='center', fontsize=10, color=TXT2)
ax.set_title('Feature Importance — Random Forest',
             fontsize=15, color=TXT, pad=14, fontweight='bold')
ax.set_xlabel('Importance (%)')
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

plt.tight_layout()
fig3.savefig('plot3_importance.png', dpi=150, bbox_inches='tight', facecolor=BG)