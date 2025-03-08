import pandas as pd

# 1) LOAD THE DATA
file_path = "University_Dataset(tmc_raw_data_2020_2029).csv"
df = pd.read_csv(file_path)

# 2) DATA INSPECTION
print("\n=== HEAD ===")
print(df.head())

print("\n=== DATAFRAME INFO ===")
df.info()

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# 3) CONVERT TIME COLUMNS TO DATETIME
#    (If count_date is purely a date with no time, you could convert that, too—but here we focus on start_time.)
df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')

# Quick check
print("\n=== PREVIEW OF DATE/TIME COLUMNS ===")
print(df[['count_date', 'start_time']].head())

# 4) CREATE HOUR AND DAY_OF_WEEK COLUMNS
df['hour'] = df['start_time'].dt.hour
df['day_of_week'] = df['start_time'].dt.day_name()

print("\n=== PREVIEW OF TIME COLUMNS (hour, day_of_week) ===")
print(df[['start_time', 'hour', 'day_of_week']].head())

# 5) AGGREGATE PEDESTRIANS & CYCLISTS
df['total_peds'] = df[['n_appr_peds','s_appr_peds','e_appr_peds','w_appr_peds']].sum(axis=1)
df['total_bikes'] = df[['n_appr_bike','s_appr_bike','e_appr_bike','w_appr_bike']].sum(axis=1)
df['total_vulnerable'] = df['total_peds'] + df['total_bikes']

# 6) AGGREGATE VEHICLES
#    (A) Cars
df['n_appr_cars'] = df[['n_appr_cars_r','n_appr_cars_t','n_appr_cars_l']].sum(axis=1)
df['s_appr_cars'] = df[['s_appr_cars_r','s_appr_cars_t','s_appr_cars_l']].sum(axis=1)
df['e_appr_cars'] = df[['e_appr_cars_r','e_appr_cars_t','e_appr_cars_l']].sum(axis=1)
df['w_appr_cars'] = df[['w_appr_cars_r','w_appr_cars_t','w_appr_cars_l']].sum(axis=1)
df['total_cars'] = df[['n_appr_cars','s_appr_cars','e_appr_cars','w_appr_cars']].sum(axis=1)

#    (B) Trucks
df['n_appr_trucks'] = df[['n_appr_truck_r','n_appr_truck_t','n_appr_truck_l']].sum(axis=1)
df['s_appr_trucks'] = df[['s_appr_truck_r','s_appr_truck_t','s_appr_truck_l']].sum(axis=1)
df['e_appr_trucks'] = df[['e_appr_truck_r','e_appr_truck_t','e_appr_truck_l']].sum(axis=1)
df['w_appr_trucks'] = df[['w_appr_truck_r','w_appr_truck_t','w_appr_truck_l']].sum(axis=1)
df['total_trucks'] = df[['n_appr_trucks','s_appr_trucks','e_appr_trucks','w_appr_trucks']].sum(axis=1)

#    (C) Buses
df['n_appr_buses'] = df[['n_appr_bus_r','n_appr_bus_t','n_appr_bus_l']].sum(axis=1)
df['s_appr_buses'] = df[['s_appr_bus_r','s_appr_bus_t','s_appr_bus_l']].sum(axis=1)
df['e_appr_buses'] = df[['e_appr_bus_r','e_appr_bus_t','e_appr_bus_l']].sum(axis=1)
df['w_appr_buses'] = df[['w_appr_bus_r','w_appr_bus_t','w_appr_bus_l']].sum(axis=1)
df['total_buses'] = df[['n_appr_buses','s_appr_buses','e_appr_buses','w_appr_buses']].sum(axis=1)

#    (D) TOTAL Vehicles
df['total_vehicles'] = df['total_cars'] + df['total_trucks'] + df['total_buses']

# 7) OPTIONAL: RENAME ANY COLUMNS YOU WANT FOR CLARITY
#    (You already renamed for cars, but we used approach-level sums above, so you can rename if you prefer.)

# 8) FINAL CHECK
print("\n=== HEAD (with new columns) ===")
print(df.head())

print("\n=== FINAL DATAFRAME INFO ===")
df.info()

print("\n=== DESCRIBE (numeric & categorical) ===")
print(df.describe(include='all'))

# Replace "University_Dataset_prepared.csv" with whatever filename you prefer
output_path = "University_Dataset_prepared.csv"
df.to_csv(output_path, index=False)
