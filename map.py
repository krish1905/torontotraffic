import pandas as pd
import folium
from folium.plugins import TimestampedGeoJson

# --- 1. LOAD AND PREPARE DATA ---

# You need a CSV with columns:
# 'start_time' (datetime), 'location_name', 'latitude', 'longitude',
# 'total_vulnerable', 'total_vehicles', etc.
df = pd.read_csv("University_Dataset_prepared.csv")

# Convert start_time to actual datetime if not done yet
df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')

# --- 2. CHOOSE THE DATE YOU WANT TO ANIMATE ---
selected_date = "2020-01-08"  # <-- Change this to any date in your dataset

# Filter to rows matching that date (year-month-day)
df_single_day = df[df['start_time'].dt.date.astype(str) == selected_date]

# If the dataset is large, you can further filter by hour range if needed:
# For example, only 6am to 10pm:
# df_single_day = df_single_day[
#     (df_single_day['start_time'].dt.hour >= 6) &
#     (df_single_day['start_time'].dt.hour <= 22)
# ]

# --- 3. GROUP BY LOCATION & HOUR ---
# Extract hour from start_time
df_single_day['hour'] = df_single_day['start_time'].dt.hour

# For each (location_name, hour), sum or average metrics as needed
grouped = df_single_day.groupby(['location_name', 'hour']).agg({
    'latitude':'mean',            # or 'first' if each location_name has consistent coords
    'longitude':'mean', 
    'total_vehicles':'sum',
    'total_vulnerable':'sum'
}).reset_index()

# --- 4. BUILD GEOJSON FEATURES ---
features = []
for _, row in grouped.iterrows():
    lat = row['latitude']
    lon = row['longitude']
    location = row['location_name']
    hour = int(row['hour'])
    
    # Create an ISO datetime string: "YYYY-MM-DDTHH:00:00"
    # (If you need minutes, you could incorporate them too.)
    time_str = f"{selected_date}T{hour:02d}:00:00"
    
    tv = int(row['total_vehicles'])
    tw = int(row['total_vulnerable'])
    
    # Scale circle radius by total_vulnerable for a quick visualization
    radius_val = 3
    if tw > 0:
        radius_val = min(20, tw / 1000.0 + 3)
        
    # Construct the GeoJSON feature
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [lon, lat]
        },
        "properties": {
            "time": time_str,  # crucial for TimestampedGeoJson
            "style": {
                "color": "blue",
                "fillColor": "blue",
                "fillOpacity": 0.6
            },
            "icon": "circle",
            "iconstyle": {
                "radius": radius_val
            },
            "popup": (
                f"<b>{location}</b><br>"
                f"Date/Hour: {selected_date} {hour:02d}:00<br>"
                f"Vehicles: {tv}<br>"
                f"Ped/Bike: {tw}"
            )
        }
    }
    features.append(feature)

geojson_data = {
    "type": "FeatureCollection",
    "features": features
}

# --- 5. CREATE THE MAP & ADD THE TIME SLIDER ---
m = folium.Map(location=[43.65, -79.38], zoom_start=12)  # Toronto center, adjust as needed

TimestampedGeoJson(
    data=geojson_data,
    transition_time=500,     # ms between frames
    loop=False,              # don't loop automatically
    auto_play=False,         # user can press 'play'
    period="PT1H",           # "Period of 1 Hour"
    add_last_point=False
).add_to(m)

# Save or just display
m.save("Single_Day_Hourly_Animation.html")

print(f"Map created for {selected_date} (hourly). Open Single_Day_Hourly_Animation.html to view.")
