# ============================================================
#   CAR PRICE PREDICTION MODEL 
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

print(" STEP 1: Libraries Imported")
print("="*55)

np.random.seed(42)
n = 1000
brands = ['Maruti','Hyundai','Honda','Toyota','Ford','BMW','Tata','Kia']
fuels = ['Petrol','Diesel','CNG','Electric']
sellers = ['Individual','Dealer','Trustmark Dealer']
transmit = ['Manual','Automatic']
owners = ['First Owner','Second Owner','Third Owner','Fourth & Above Owner']
brand_base = {'Maruti':4.5,'Hyundai':6.5,'Honda':8.0,'Toyota':10.0,
              'Ford':7.5,'BMW':35.0,'Tata':7.0,'Kia':9.5}

rows = []
for _ in range(n):
    brand = np.random.choice(brands, p=[0.25,0.20,0.15,0.10,0.10,0.05,0.10,0.05])
    year = int(np.random.randint(2010, 2024))
    km = int(np.random.randint(5000, 180000))
    fuel = np.random.choice(fuels, p=[0.45,0.35,0.12,0.08])
    seller = np.random.choice(sellers)
    trans = np.random.choice(transmit, p=[0.65,0.35])
    owner = np.random.choice(owners, p=[0.55,0.28,0.12,0.05])
    engine = int(np.random.choice([800,1000,1200,1500,1800,2000,2500,3000],
                                   p=[0.05,0.10,0.20,0.25,0.15,0.12,0.08,0.05]))
    power = round(float(engine * np.random.uniform(0.06, 0.10)), 1)
    seats = int(np.random.choice([4,5,6,7], p=[0.05,0.70,0.10,0.15]))
    age = 2024 - year
    depr = max(0.25, 1 - age * 0.07)
    km_pen = max(0.60, 1 - (km / 180000) * 0.40)
    fuel_m = {'Petrol':1.0,'Diesel':1.08,'CNG':0.88,'Electric':1.30}[fuel]
    own_m = {'First Owner':1.0,'Second Owner':0.87,
               'Third Owner':0.75,'Fourth & Above Owner':0.62}[owner]
    tr_m = 1.10 if trans == 'Automatic' else 1.0
    noise = float(np.random.normal(1.0, 0.07))
    price = round(max(0.50, brand_base[brand]*depr*km_pen*fuel_m*own_m*tr_m*noise), 2)
    rows.append([brand,year,km,fuel,seller,trans,owner,engine,power,seats,price])

df = pd.DataFrame(rows, columns=[
    'brand','year','km_driven','fuel','seller_type','transmission',
    'owner','engine','max_power','seats','selling_price'])

rng = np.random.default_rng(42)
for col in ['km_driven','engine','max_power']:
    idx = rng.choice(n, size=int(n*0.08), replace=False)
    df.loc[idx, col] = np.nan

print(" STEP 2: Dataset Created")
print(f" Rows: {df.shape[0]}, Columns: {df.shape[1]}")
missing = df.isnull().sum()
print(f" Missing values: km_driven={missing['km_driven']}, engine={missing['engine']}, max_power={missing['max_power']}")
print("="*55)

print("\n STEP 3: Data Cleaning & Preprocessing")
df2 = df.copy()

for col in ['km_driven','engine','max_power']:
    med = df2[col].median()
    df2[col] = df2[col].fillna(med)
    df2[col] = df2[col].astype(float)

print(" Missing values → filled with median")

df2['car_age'] = (2024 - df2['year']).astype(float)
df2['km_per_year'] = (df2['km_driven'] / (df2['car_age'] + 1)).round(1)
print(" car_age, km_per_year engineered")

le_dict = {}
for col in ['brand','fuel','seller_type','transmission','owner']:
    le = LabelEncoder()
    df2[col+'_enc'] = le.fit_transform(df2[col].astype(str))
    le_dict[col] = le
print(" Label encoding done")

features = ['car_age','km_driven','km_per_year','engine','max_power',
            'seats','brand_enc','fuel_enc','transmission_enc','owner_enc']
X = df2[features].copy().astype(float)
y = df2['selling_price'].copy().astype(float)

print(f" NaN check X: {X.isnull().sum().sum()}, y: {y.isnull().sum()}")
assert X.isnull().sum().sum() == 0, "NaN in X!"
assert y.isnull().sum() == 0, "NaN in y!"

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
print(f" StandardScaler applied")
print(f"\n STEP 3 DONE : X shape: {X_scaled.shape}")
print("="*55)

X_train,X_test,y_train,y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)
print(f"\n STEP 4: Train/Test Split")
print(f" Train: {len(X_train)}, Test: {len(X_test)}")
print("="*55)

print("\n STEP 5: Training")
lr = LinearRegression()
lr.fit(X_train, y_train)
print(" Linear Regression trained")

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print(" Random Forest (100 trees) trained")
print(" STEP 5 DONE!")
print("="*55)

print("\n STEP 6: Evaluation")
def evaluate(name, model, Xt, yt):
    p = np.maximum(model.predict(Xt), 0)
    mae = mean_absolute_error(yt, p)
    rmse = np.sqrt(mean_squared_error(yt, p))
    r2 = r2_score(yt, p)
    print(f"\n    {name}")
    print(f" MAE : ₹{mae:.2f} Lakhs")
    print(f" RMSE : ₹{rmse:.2f} Lakhs")
    print(f" R² : {r2:.4f} ({r2*100:.1f}% variance explained)")
    return p, mae, rmse, r2

lr_p, lr_mae, lr_rmse, lr_r2 = evaluate("Linear Regression", lr, X_test, y_test)
rf_p, rf_mae, rf_rmse, rf_r2 = evaluate("Random Forest", rf, X_test, y_test)
print(f"\n Winner: {'Random Forest' if rf_r2>lr_r2 else 'Linear Regression'}")
print("="*55)

print("\n STEP 7: Predictions on New Cars")
def predict_price(brand,year,km,fuel,seller,trans,owner_t,eng,pwr,seats_n):
    age = float(2024 - year)
    km_yr = float(km) / (age + 1)
    b = int(le_dict['brand'].transform([brand])[0])
    f = int(le_dict['fuel'].transform([fuel])[0])
    s = int(le_dict['seller_type'].transform([seller])[0])
    t = int(le_dict['transmission'].transform([trans])[0])
    o = int(le_dict['owner'].transform([owner_t])[0])
    row = pd.DataFrame([[age,float(km),km_yr,float(eng),float(pwr),float(seats_n),b,f,t,o]], columns=features)
    row_sc = pd.DataFrame(scaler.transform(row.astype(float)), columns=features)
    lrp = max(0,float(lr.predict(row_sc)[0]))
    rfp = max(0,float(rf.predict(row_sc)[0]))
    print(f" {year} {brand} {fuel} {trans} {km:,} km {owner_t}")
    print(f" Linear Regression ₹{lrp:.2f}L, Random Forest ₹{rfp:.2f}L ")
    print()
    return rfp

predict_price('Maruti',2020,30000,'Petrol','Individual','Manual','First Owner',1200,82.0, 5)
predict_price('Hyundai',2019,55000,'Diesel','Dealer','Manual','Second Owner',1500,115.0,7)
predict_price('BMW',2021,20000,'Petrol','Dealer','Automatic','First Owner',2000,190.0,5)
predict_price('Maruti',2014,90000,'CNG','Individual','Manual','Third Owner',998,67.0, 5)

imp   = rf.feature_importances_
fi_df = pd.DataFrame({'Feature':features,'Importance':imp}).sort_values('Importance',ascending=False)
print(" Feature Importances:")
for _, row in fi_df.iterrows():
    bar = '█' * int(row['Importance']*80)
    print(f" {row['Feature']:22s} {bar} {row['Importance']*100:.1f}%")

print("\n ALL 7 STEPS COMPLETE!")
print("="*55)

with open('model_data.pkl','wb') as f:
    pickle.dump({'df':df,'df2':df2,'X_test':X_test,'y_test':y_test,
                 'lr_p':lr_p,'rf_p':rf_p,'lr_r2':lr_r2,'rf_r2':rf_r2,
                 'lr_mae':lr_mae,'rf_mae':rf_mae,'lr_rmse':lr_rmse,'rf_rmse':rf_rmse,
                 'fi_df':fi_df,'rf':rf,'lr':lr,'features':features,
                 'scaler':scaler,'le_dict':le_dict}, f)
print("Model data saved.")
