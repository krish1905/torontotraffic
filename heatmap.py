import pandas as pd
import folium
from folium.plugins import HeatMap

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')


# Load your cleaned dataset
df = pd.read_csv("cleaned_traffic_data.csv")

# Create 'total_peds' by summing pedestrian activity across all directions
df["total_peds"] = df["n_appr_peds"] + df["s_appr_peds"] + df["e_appr_peds"] + df["w_appr_peds"]

# Ensure latitude and longitude are present for creating the map
df = df.dropna(subset=["latitude", "longitude"])

# Group by intersection (location_name) and sum pedestrian activity
pedestrian_activity = df.groupby("location_name")["total_peds"].sum()

# Find the top 10 intersections with the highest pedestrian activity
top_pedestrian_locations = pedestrian_activity.nlargest(10)

# Create a base map centered on Toronto (adjust coordinates as needed)
m = folium.Map(location=[43.7, -79.42], zoom_start=12)

# Prepare data for heatmap, using latitude, longitude, and total pedestrian activity
heat_map_data = []
for index, row in df.iterrows():
    if row["location_name"] in top_pedestrian_locations.index:
        heat_map_data.append([row['latitude'], row['longitude'], row['total_peds']])

# Add heatmap layer to the map
HeatMap(heat_map_data).add_to(m)

# Save the map to an HTML file
m.save("pedestrian_heatmap.html")

# Show the map in the notebook (or open it in the browser)
m
