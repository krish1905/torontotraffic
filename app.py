from flask import Flask, render_template, jsonify, request, send_file
import pandas as pd
import json
import folium
from folium.plugins import HeatMap, MarkerCluster, FastMarkerCluster
import numpy as np
from datetime import datetime
import os
import branca.colormap as cm
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure OpenAI
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("Warning: OpenAI API key not found in environment variables")
try:
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.openai.com/v1"
    )
except Exception as e:
    print(f"Warning: Could not initialize OpenAI client: {e}")
    client = None

app = Flask(__name__)

# Load and preprocess the datasets
print("Loading datasets...")
try:
    # Read the CSV file
    df = pd.read_csv("University_Dataset_prepared.csv")
    print(f"Loaded {len(df)} records")
    
    # Print the first few rows to verify data
    print("\nFirst few rows of data:")
    print(df[['latitude', 'longitude', 'n_appr_peds', 's_appr_peds', 'e_appr_peds', 'w_appr_peds']].head())
    
    # Calculate totals for each type
    df['total_pedestrians'] = df[['n_appr_peds', 's_appr_peds', 'e_appr_peds', 'w_appr_peds']].sum(axis=1)
    df['total_bikes'] = df[['n_appr_bike', 's_appr_bike', 'e_appr_bike', 'w_appr_bike']].sum(axis=1)
    df['total_vehicles'] = df[['n_appr_cars', 's_appr_cars', 'e_appr_cars', 'w_appr_cars']].sum(axis=1)
    
    # Calculate safety scores and other metrics
    df['total_traffic'] = df['total_pedestrians'] + df['total_bikes'] + df['total_vehicles']
    df['vulnerable_users'] = df['total_pedestrians'] + df['total_bikes']
    df['safety_score'] = 10 * (df['vulnerable_users'] / df['total_traffic'].where(df['total_traffic'] > 0, 1))
    df['vehicle_ratio'] = df['total_vehicles'] / df['total_traffic'].where(df['total_traffic'] > 0, 1)
    
    print("\nData summary:")
    print(f"Total pedestrians: {df['total_pedestrians'].sum():,}")
    print(f"Total bikes: {df['total_bikes'].sum():,}")
    print(f"Total vehicles: {df['total_vehicles'].sum():,}")
    
    print("Data preprocessing complete.")

except Exception as e:
    print(f"Error during data loading/preprocessing: {str(e)}")
    df = pd.DataFrame()

def create_visualization(data_type='pedestrians', viz_type='heatmap'):
    # Create a base map centered on Toronto
    m = folium.Map(location=[43.7, -79.42], zoom_start=12, tiles='cartodbpositron')
    
    # Get the values based on data type
    if data_type == 'pedestrians':
        values = df['total_pedestrians']
        color = 'YlOrRd'
        title = 'Pedestrian Activity'
        icon = 'user'
    elif data_type == 'bikes':
        values = df['total_bikes']
        color = 'YlGn'
        title = 'Bicycle Activity'
        icon = 'bicycle'
    else:
        values = df['total_vehicles']
        color = 'PuBu'
        title = 'Vehicle Traffic'
        icon = 'car'

    # Create visualization based on type
    if viz_type == 'heatmap':
        # Create heatmap data following map.py approach
        heat_data = [[row['latitude'], row['longitude'], float(val)] 
                    for _, row in df.iterrows() 
                    for val in [values.iloc[_]]
                    if pd.notna(row['latitude']) and pd.notna(row['longitude']) and pd.notna(val)]
        
        # Add heatmap layer
        HeatMap(heat_data).add_to(m)

    elif viz_type == 'markers':
        # Create marker cluster
        marker_cluster = MarkerCluster(name=title).add_to(m)
        
        # Add markers for top locations
        top_locations = df.nlargest(50, values.name)
        for _, row in top_locations.iterrows():
            popup_html = f"""
                <div style="font-family: Arial; width: 200px;">
                    <h4>{row.get('location_name', 'Intersection')}</h4>
                    <p><b>Pedestrians:</b> {int(row['total_pedestrians']):,}</p>
                    <p><b>Bicycles:</b> {int(row['total_bikes']):,}</p>
                    <p><b>Vehicles:</b> {int(row['total_vehicles']):,}</p>
                </div>
            """
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=popup_html,
                icon=folium.Icon(color='red' if values.iloc[_] > values.mean() + values.std() else 'blue', 
                                icon=icon, prefix='fa')
            ).add_to(marker_cluster)

    elif viz_type == 'circles':
        # Create a colormap
        colormap = cm.LinearColormap(
            colors=['green', 'yellow', 'red'],
            vmin=values.min(),
            vmax=values.max()
        )
        m.add_child(colormap)
        
        # Add circle markers
        for _, row in df.iterrows():
            val = values.iloc[_]
            if pd.notna(val) and pd.notna(row['latitude']) and pd.notna(row['longitude']):
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=np.sqrt(val) / 5,  # Adjusted radius scaling
                    popup=f"{title}: {int(val):,}",
                    color=colormap(val),
                    fill=True,
                    fill_color=colormap(val)
                ).add_to(m)

    elif viz_type == 'choropleth':
        # Create clusters of nearby points and calculate average values
        from sklearn.cluster import DBSCAN
        
        # Prepare data for clustering
        coords = df[['latitude', 'longitude']].values
        clustering = DBSCAN(eps=0.01, min_samples=2).fit(coords)
        df['cluster'] = clustering.labels_
        
        # Calculate cluster statistics
        cluster_stats = df[df['cluster'] >= 0].groupby('cluster').agg({
            'latitude': 'mean',
            'longitude': 'mean',
            values.name: 'mean'
        })
        
        # Create a colormap
        colormap = cm.LinearColormap(
            colors=['green', 'yellow', 'red'],
            vmin=cluster_stats[values.name].min(),
            vmax=cluster_stats[values.name].max()
        )
        m.add_child(colormap)
        
        # Add polygons for each cluster
        for cluster_id, stats in cluster_stats.iterrows():
            # Create a circular polygon around the cluster center
            center = [stats['latitude'], stats['longitude']]
            points = []
            for angle in range(0, 360, 10):
                points.append([
                    center[0] + 0.005 * np.cos(np.radians(angle)),
                    center[1] + 0.005 * np.sin(np.radians(angle))
                ])
            
            folium.Polygon(
                locations=points,
                popup=f"{title}: {int(stats[values.name]):,}",
                color=colormap(stats[values.name]),
                fill=True,
                fill_color=colormap(stats[values.name]),
                fill_opacity=0.7
            ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save the map
    map_file = f'static/{data_type}_{viz_type}.html'
    m.save(map_file)
    return map_file

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_visualization')
def get_visualization():
    data_type = request.args.get('type', 'pedestrians')
    viz_type = request.args.get('viz', 'heatmap')
    
    if data_type not in ['pedestrians', 'bikes', 'vehicles']:
        return jsonify({'error': 'Invalid data type'}), 400
    
    if viz_type not in ['heatmap', 'markers', 'circles', 'choropleth']:
        return jsonify({'error': 'Invalid visualization type'}), 400
    
    try:
        map_file = create_visualization(data_type, viz_type)
        return send_file(map_file)
    except Exception as e:
        print(f"Error creating visualization: {e}")
        return jsonify({'error': 'Failed to create visualization'}), 500

@app.route('/get_intersection_data')
def get_intersection_data():
    try:
        lat = float(request.args.get('lat'))
        lng = float(request.args.get('lng'))
        
        if df.empty:
            return jsonify({'error': 'No data available'}), 404
        
        # Find nearest intersection
        df['distance'] = ((df['latitude'] - lat)**2 + (df['longitude'] - lng)**2)**0.5
        nearest = df.loc[df['distance'].idxmin()]
        
        response = {
            'name': nearest.get('location_name', f"Intersection at ({nearest['latitude']:.4f}, {nearest['longitude']:.4f})"),
            'pedestrian_count': int(nearest['total_pedestrians']),
            'bike_count': int(nearest['total_bikes']),
            'vehicle_count': int(nearest['total_vehicles']),
            'safety_score': float(nearest['safety_score'])
        }
        
        return jsonify(response)
    except Exception as e:
        print(f"Error in get_intersection_data: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/ask_ai')
def ask_ai():
    try:
        if not client:
            return jsonify({'error': 'AI service not available'}), 503
            
        question = request.args.get('question')
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        if df.empty:
            return jsonify({'error': 'No data available to answer questions'}), 404

        try:
            # Prepare context about the data
            context = {
                'total_intersections': len(df),
                'total_pedestrians': int(df['total_pedestrians'].sum()),
                'total_bikes': int(df['total_bikes'].sum()),
                'total_vehicles': int(df['total_vehicles'].sum())
            }

            # Create prompt for OpenAI
            prompt = f"""You are a traffic safety expert analyzing Toronto intersection data. 
            Here's the current data summary:
            - Analyzing {context['total_intersections']} intersections
            - Total pedestrian count: {context['total_pedestrians']:,}
            - Total bicycle count: {context['total_bikes']:,}
            - Total vehicle count: {context['total_vehicles']:,}
            
            Question: {question}
            
            Please provide a clear, concise answer based on the data provided."""

            # Get response from OpenAI
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful traffic safety expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )

            answer = response.choices[0].message.content
            return jsonify({'answer': answer})

        except Exception as e:
            print(f"Error processing OpenAI request: {str(e)}")
            return jsonify({'error': 'Failed to process AI request'}), 500

    except Exception as e:
        print(f"Error in ask_ai: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create static folder if it doesn't exist
    os.makedirs('static', exist_ok=True)
    app.run(debug=True) 