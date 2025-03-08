import pandas as pd
import matplotlib.pyplot as plt

# 1) LOAD PREPARED DATA
df = pd.read_csv("University_Dataset_prepared.csv")

# Quick look (optional)
print(df.head())
print(df.info())

##################################
# 2) AGGREGATE BY LOCATION (INTERSECTION)
##################################
location_summary = df.groupby('location_name').agg({
    'total_vulnerable': 'sum',
    'total_vehicles': 'sum'
}).reset_index()

# 2.1. Top 10 by total_vulnerable
top_vulnerable = location_summary.nlargest(10, 'total_vulnerable')

# 2.2. Create a ratio for vulnerable_to_vehicle
location_summary['vulnerable_vehicle_ratio'] = (
    location_summary['total_vulnerable'] / location_summary['total_vehicles'].replace(0, 1)
)

# 2.3. Top 10 by ratio
top_ratio = location_summary.nlargest(10, 'vulnerable_vehicle_ratio')

##################################
# 3) VISUALIZE TOP INTERSECTIONS
##################################

# Print the data behind the "Top 10 by total_vulnerable" bar chart
print("\n=== Data: Top 10 Intersections by Total Vulnerable ===")
print(top_vulnerable)

# 3.1. Bar Chart: Top 10 Intersections by Total Vulnerable
plt.figure()
plt.bar(top_vulnerable['location_name'], top_vulnerable['total_vulnerable'])
plt.title("Top 10 Intersections by Total Ped/Bike (Vulnerable) Volume")
plt.xticks(rotation=90)
plt.ylabel("Total Vulnerable Users")
plt.tight_layout()  # Helps ensure x-labels fit
plt.savefig("top_10_intersections_vulnerable.png")
plt.show()

# Print the data behind the "Top 10 by vulnerable/vehicle ratio" bar chart
print("\n=== Data: Top 10 Intersections by Vulnerable-to-Vehicle Ratio ===")
print(top_ratio)

# 3.2. Bar Chart: Top 10 Intersections by Vulnerable/Vehicle Ratio
plt.figure()
plt.bar(top_ratio['location_name'], top_ratio['vulnerable_vehicle_ratio'])
plt.title("Top 10 Intersections by Vulnerable-to-Vehicle Ratio")
plt.xticks(rotation=90)
plt.ylabel("Ratio (Ped+Bikes / Vehicles)")
plt.tight_layout()
plt.savefig("top_10_intersections_ratio.png")
plt.show()

##################################
# 4) TIME-BASED ANALYSIS
##################################

# 4.1. By Hour of Day
hourly_summary = df.groupby('hour').agg({
    'total_vulnerable':'sum',
    'total_vehicles':'sum'
}).reset_index()

# Print the data behind the "Vulnerable by Hour" bar chart
print("\n=== Data: Hourly Summary (Total Vulnerable & Vehicles) ===")
print(hourly_summary)

# Bar Chart: Vulnerable by Hour
plt.figure()
plt.bar(hourly_summary['hour'], hourly_summary['total_vulnerable'])
plt.title("Total Vulnerable Road Users by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Total Vulnerable Users")
plt.tight_layout()
plt.savefig("hourly_vulnerable.png")
plt.show()

# 4.2. By Day of Week
day_summary = df.groupby('day_of_week').agg({
    'total_vulnerable':'sum',
    'total_vehicles':'sum'
}).reset_index()

# Print the data behind the "Vulnerable by Day of Week" bar chart
print("\n=== Data: Day of Week Summary (Total Vulnerable & Vehicles) ===")
print(day_summary)

# Bar Chart: Vulnerable by Day of Week
plt.figure()
plt.bar(day_summary['day_of_week'], day_summary['total_vulnerable'])
plt.title("Total Vulnerable Road Users by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Total Vulnerable Users")
plt.tight_layout()
plt.savefig("day_of_week_vulnerable.png")
plt.show()

# (Optional) Bar Chart: Total vehicles by Day of Week
plt.figure()
plt.bar(day_summary['day_of_week'], day_summary['total_vehicles'])
plt.title("Total Vehicles by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Total Vehicles")
plt.tight_layout()
plt.savefig("day_of_week_vehicles.png")
plt.show()

##################################
# 5) OPTIONAL EXTRAS
##################################

# A. Save the aggregated location summary (if you want to share it)
location_summary.to_csv("Intersection_Safety_Summary.csv", index=False)

# B. Inspect top locations (already printed above, but you can keep or remove)
print("\n=== Top 10 Intersections by Total Vulnerable (Printed Again) ===")
print(top_vulnerable)

print("\n=== Top 10 Intersections by Vulnerable-to-Vehicle Ratio (Printed Again) ===")
print(top_ratio)
