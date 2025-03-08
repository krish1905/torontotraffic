import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("cleaned_traffic_data.csv")  # Replace with actual filename

# Aggregate pedestrian and cyclist data by location
df["total_peds"] = df["n_appr_peds"] + df["s_appr_peds"] + df["e_appr_peds"] + df["w_appr_peds"]
df["total_bikes"] = df["n_appr_bike"] + df["s_appr_bike"] + df["e_appr_bike"] + df["w_appr_bike"]

# Filter locations with the highest pedestrian and cyclist activity
top_pedestrian_locations = df.groupby("location_name")["total_peds"].sum().nlargest(10)
top_bike_locations = df.groupby("location_name")["total_bikes"].sum().nlargest(10)

# Plot high-risk pedestrian locations
plt.figure(figsize=(10, 5))
top_pedestrian_locations.plot(kind="bar", color="red")
plt.title("Top 10 Locations with Highest Pedestrian Activity")
plt.ylabel("Total Pedestrians")
plt.xlabel("Location")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Plot high-risk cyclist locations
plt.figure(figsize=(10, 5))
top_bike_locations.plot(kind="bar", color="blue")
plt.title("Top 10 Locations with Highest Bicycle Activity")
plt.ylabel("Total Bicycles")
plt.xlabel("Location")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

df["hour"] = pd.to_datetime(df["start_time"]).dt.hour  # Extract hour from time

# Aggregate total pedestrians and bikes per hour
hourly_pedestrian_activity = df.groupby("hour")["total_peds"].sum()
hourly_cyclist_activity = df.groupby("hour")["total_bikes"].sum()

# Plot
plt.figure(figsize=(10, 5))
plt.plot(hourly_pedestrian_activity, marker='o', label="Pedestrians", color="red")
plt.plot(hourly_cyclist_activity, marker='o', label="Cyclists", color="blue")
plt.xlabel("Hour of the Day")
plt.ylabel("Total Activity")
plt.title("Pedestrian & Cyclist Activity by Hour")
plt.legend()
plt.grid()
plt.show()

