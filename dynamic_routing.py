#!/usr/bin/env python
"""
Dynamic Ship Routing System
Optimizes maritime routes while avoiding land masses using ML approaches and AIS data
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import requests
import datetime
import math
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from matplotlib import colors
import heapq  # For A* algorithm

# Configuration
CONFIG = {
    "api_key": "AABBCCDD",  # Replace with actual API key
    "base_url": "https://api.vtexplorer.com/",
    "satellite_data_path": "data/water_occurrence.geojson",
    "ais_data_cache": "data/ais_cache/",
    "models_path": "models/",
    "land_detection_threshold": 0.3,  # Water probability threshold
}

class TerrainRecognition:
    """Handles land/water detection using satellite data"""
    
    def __init__(self, config):
        self.config = config
        self.water_data = self.load_water_data()
        
    def load_water_data(self):
        """Load water occurrence data from GeoJSON file"""
        try:
            with open(self.config['satellite_data_path'], 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Water occurrence data not found at {self.config['satellite_data_path']}")
            print("Using simplified land mass detection...")
            return self.generate_simplified_data()
    
    def generate_simplified_data(self):
        """Generate simplified land/water data for major water bodies and land masses"""
        # Simplified world map with major landmasses
        return {
            "type": "FeatureCollection",
            "features": [
                # Major continents and landmasses
                self.create_landmass_feature("Asia", [
                    [35, 70], [35, 145], [75, 145], [75, 70]
                ]),
                self.create_landmass_feature("Europe", [
                    [35, -10], [35, 40], [70, 40], [70, -10]
                ]),
                self.create_landmass_feature("Africa", [
                    [-35, -20], [-35, 50], [35, 50], [35, -20]
                ]),
                self.create_landmass_feature("North America", [
                    [15, -170], [15, -50], [75, -50], [75, -170]
                ]),
                self.create_landmass_feature("South America", [
                    [-55, -80], [-55, -35], [10, -35], [10, -80]
                ]),
                self.create_landmass_feature("Australia", [
                    [-40, 110], [-40, 155], [-10, 155], [-10, 110]
                ]),
                self.create_landmass_feature("Southeast Asia", [
                    [0, 95], [0, 140], [20, 140], [20, 95]
                ]),
                self.create_landmass_feature("India", [
                    [5, 70], [5, 90], [30, 90], [30, 70]
                ]),
                # Major islands
                self.create_landmass_feature("Indonesia", [
                    [-10, 95], [-10, 140], [10, 140], [10, 95]
                ])
            ]
        }
    
    def create_landmass_feature(self, name, coordinates):
        """Helper to create a GeoJSON landmass feature"""
        return {
            "type": "Feature",
            "properties": {"name": name, "is_land": True},
            "geometry": {
                "type": "Polygon",
                "coordinates": [coordinates + [coordinates[0]]]  # Close the polygon
            }
        }
    
    def is_land(self, lat, lon):
        """Check if a given coordinate is on land based on satellite data"""
        # Check against known landmasses in simplified data
        for feature in self.water_data["features"]:
            if feature["properties"].get("is_land", False):
                if self.point_in_polygon(lat, lon, feature["geometry"]["coordinates"][0]):
                    return True
        
        # Also check for specific regions that need more detailed handling
        # Indian subcontinent
        if (lat > 5 and lat < 35 and lon > 65 and lon < 90):
            return self.indian_landmass_check(lat, lon)
        
        # Southeast Asia
        if (lat > -10 and lat < 25 and lon > 95 and lon < 140):
            return self.southeast_asia_check(lat, lon)
            
        return False
    
    def indian_landmass_check(self, lat, lon):
        """Detailed check for Indian subcontinent"""
        # Main Indian peninsula
        if (lat > 5 and lat < 30):
            # West coast (Arabian Sea)
            if lon > 69 and lon < 76:
                return True
            # East coast (Bay of Bengal)
            if lon > 80 and lon < 90:
                return True
            # Central India
            if lon > 75 and lon < 85 and lat > 10 and lat < 25:
                return True
            # Sri Lanka
            if lon > 79 and lon < 82 and lat > 5 and lat < 10:
                return True
        return False
    
    def southeast_asia_check(self, lat, lon):
        """Detailed check for Southeast Asia"""
        # Thailand/Malaysia/Vietnam mainland
        if lat > 5 and lat < 25 and lon > 97 and lon < 110:
            return True
        
        # Indonesian archipelago
        if lat > -10 and lat < 5:
            # Sumatra
            if lon > 95 and lon < 108:
                return True
            # Java
            if lon > 105 and lon < 115 and lat > -9 and lat < -5:
                return True
            # Borneo
            if lon > 109 and lon < 119 and lat > -5 and lat < 7:
                return True
        
        # Philippines
        if lat > 5 and lat < 20 and lon > 117 and lon < 126:
            return True
            
        return False
    
    def point_in_polygon(self, lat, lon, polygon):
        """Check if point (lat, lon) is inside a polygon using ray casting algorithm"""
        inside = False
        j = len(polygon) - 1
        
        for i in range(len(polygon)):
            # Check if the point is inside the polygon
            if ((polygon[i][1] > lat) != (polygon[j][1] > lat)) and \
               (lon < polygon[i][0] + (polygon[j][0] - polygon[i][0]) * 
                (lat - polygon[i][1]) / (polygon[j][1] - polygon[i][1])):
                inside = not inside
            j = i
            
        return inside
    
    def water_probability(self, lat, lon):
        """Return probability that a location is water (0 = land, 1 = water)"""
        return 0.0 if self.is_land(lat, lon) else 1.0

class AISDataIntegration:
    """Handles AIS data integration for real-time ship tracking"""
    
    def __init__(self, config):
        self.config = config
        self.api_key = config["api_key"]
        self.base_url = config["base_url"]
        self.cache_dir = config["ais_data_cache"]
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_live_data(self, lat, lon, radius=100):
        """Get live AIS data for ships in area"""
        endpoint = "livedata"
        params = {
            "userkey": self.api_key,
            "lat": lat,
            "lon": lon,
            "radius": radius,  # in nautical miles
            "format": "json"
        }
        
        cache_file = os.path.join(
            self.cache_dir, 
            f"live_{lat}_{lon}_{radius}_{datetime.datetime.now().strftime('%Y%m%d_%H')}.json"
        )
        
        # Check cache first (hourly caching)
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        # If not in cache, get from API
        try:
            response = requests.get(f"{self.base_url}{endpoint}", params=params)
            response.raise_for_status()
            data = response.json()
            
            # Cache the result
            with open(cache_file, 'w') as f:
                json.dump(data, f)
                
            return data
            
        except Exception as e:
            print(f"Error fetching AIS data: {e}")
            # Return empty data structure
            return {"vessels": []}
    
    def get_vessel_information(self, mmsi):
        """Get detailed information about a specific vessel by MMSI"""
        endpoint = "vessels"
        params = {
            "userkey": self.api_key,
            "mmsi": mmsi,
            "format": "json"
        }
        
        cache_file = os.path.join(self.cache_dir, f"vessel_{mmsi}.json")
        
        # Check cache first (this data doesn't change often)
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        # If not in cache, get from API
        try:
            response = requests.get(f"{self.base_url}{endpoint}", params=params)
            response.raise_for_status()
            data = response.json()
            
            # Cache the result
            with open(cache_file, 'w') as f:
                json.dump(data, f)
                
            return data
            
        except Exception as e:
            print(f"Error fetching vessel information: {e}")
            return {"vessel": None}
    
    def get_traffic_density(self, route_waypoints, radius=10):
        """Get traffic density along a proposed route"""
        density_data = []
        
        # Sample every 5th waypoint to reduce API calls
        sampled_waypoints = route_waypoints[::5]
        
        for wp in sampled_waypoints:
            lat, lon = wp
            live_data = self.get_live_data(lat, lon, radius)
            vessel_count = len(live_data.get("vessels", []))
            density_data.append({
                "waypoint": wp,
                "vessel_count": vessel_count,
                "timestamp": datetime.datetime.now().isoformat()
            })
        
        return density_data
    
    def analyze_historical_routes(self, start_lat, start_lon, end_lat, end_lon):
        """Analyze historical routes between two points"""
        # This would normally use the AIS historical data API
        # For this implementation, we'll return a simulated result
        
        # Simulate common routes between these points
        return {
            "route_count": 25,
            "common_waypoints": [
                [start_lat + (end_lat - start_lat) * 0.25, start_lon + (end_lon - start_lon) * 0.25],
                [start_lat + (end_lat - start_lat) * 0.5, start_lon + (end_lon - start_lon) * 0.5],
                [start_lat + (end_lat - start_lat) * 0.75, start_lon + (end_lon - start_lon) * 0.75]
            ],
            "avg_transit_time": 120  # hours
        }

class PathfindingAlgorithms:
    """Implements pathfinding algorithms for ship routing"""
    
    def __init__(self, terrain_recognition):
        self.terrain = terrain_recognition
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate haversine distance between two points in nautical miles"""
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371.0 / 1.852  # Earth radius in km converted to nautical miles
        return c * r
    
    def astar_distance(self, a, b):
        """A* heuristic function - straight line distance"""
        return self.haversine_distance(a[0], a[1], b[0], b[1])
    
    def a_star_pathfinding(self, start, goal, resolution=0.5):
        """A* pathfinding algorithm for maritime routing
        
        Args:
            start: [lat, lon] starting point
            goal: [lat, lon] destination point
            resolution: Grid resolution in degrees
            
        Returns:
            List of waypoints forming the optimal path
        """
        # Create a grid around the start and goal
        start_lat, start_lon = start
        goal_lat, goal_lon = goal
        
        # Define the search area with padding
        min_lat = min(start_lat, goal_lat) - 5
        max_lat = max(start_lat, goal_lat) + 5
        min_lon = min(start_lon, goal_lon) - 5
        max_lon = max(start_lon, goal_lon) + 5
        
        # Create a simple grid representation
        grid_rows = int((max_lat - min_lat) / resolution)
        grid_cols = int((max_lon - min_lon) / resolution)
        
        # Helper function to convert lat/lon to grid coordinates
        def to_grid(lat, lon):
            row = int((lat - min_lat) / resolution)
            col = int((lon - min_lon) / resolution)
            return (row, col)
        
        # Helper function to convert grid coordinates to lat/lon
        def to_latlon(row, col):
            lat = min_lat + row * resolution
            lon = min_lon + col * resolution
            return (lat, lon)
        
        # Convert start and goal to grid coordinates
        start_grid = to_grid(start_lat, start_lon)
        goal_grid = to_grid(goal_lat, goal_lon)
        
        # Define valid moves (8 directions)
        moves = [
            (-1, 0),  # North
            (1, 0),   # South
            (0, 1),   # East
            (0, -1),  # West
            (-1, 1),  # Northeast
            (1, 1),   # Southeast
            (1, -1),  # Southwest
            (-1, -1)  # Northwest
        ]
        
        # Initialize A* algorithm
        open_set = []
        heapq.heappush(open_set, (0, start_grid))  # (priority, position)
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.astar_distance(start, goal)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal_grid:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(to_latlon(*current))
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            for move in moves:
                neighbor = (current[0] + move[0], current[1] + move[1])
                
                # Check if neighbor is valid (within grid)
                if 0 <= neighbor[0] < grid_rows and 0 <= neighbor[1] < grid_cols:
                    # Check if neighbor is on water
                    neighbor_latlon = to_latlon(*neighbor)
                    if not self.terrain.is_land(neighbor_latlon[0], neighbor_latlon[1]):
                        # Valid neighbor, calculate tentative g_score
                        tentative_g = g_score.get(current, float('inf')) + \
                                      self.haversine_distance(
                                          to_latlon(*current)[0], to_latlon(*current)[1],
                                          neighbor_latlon[0], neighbor_latlon[1]
                                      )
                        
                        if tentative_g < g_score.get(neighbor, float('inf')):
                            # This path is better than any previous one
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g
                            f_score[neighbor] = tentative_g + self.astar_distance(
                                neighbor_latlon, goal
                            )
                            
                            if neighbor not in [i[1] for i in open_set]:
                                heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # If we get here, no path was found
        print("No path found! Using direct route with waypoints.")
        
        # Return a direct route with intermediate waypoints
        return self.create_direct_route_with_waypoints(start, goal)
    
    def create_direct_route_with_waypoints(self, start, goal, num_waypoints=10):
        """Create a direct route with intermediate waypoints, adjusted to avoid land"""
        route = [start]
        
        start_lat, start_lon = start
        goal_lat, goal_lon = goal
        
        for i in range(1, num_waypoints + 1):
            # Create waypoint by linear interpolation
            ratio = i / (num_waypoints + 1)
            wp_lat = start_lat + (goal_lat - start_lat) * ratio
            wp_lon = start_lon + (goal_lon - start_lon) * ratio
            
            # Check if waypoint is on land and adjust if needed
            attempts = 0
            while self.terrain.is_land(wp_lat, wp_lon) and attempts < 10:
                # Adjust waypoint outward from the direct line
                perpendicular_angle = math.atan2(goal_lat - start_lat, goal_lon - start_lon) + math.pi/2
                adjustment = 0.5 * (attempts + 1)  # Increase adjustment with each attempt
                
                # Try different directions if needed
                if attempts % 2 == 0:
                    wp_lat += adjustment * math.sin(perpendicular_angle)
                    wp_lon += adjustment * math.cos(perpendicular_angle)
                else:
                    wp_lat -= adjustment * math.sin(perpendicular_angle)
                    wp_lon -= adjustment * math.cos(perpendicular_angle)
                
                attempts += 1
            
            # Add waypoint if not on land, otherwise skip
            if not self.terrain.is_land(wp_lat, wp_lon):
                route.append([wp_lat, wp_lon])
        
        route.append(goal)
        return route
    
    def optimize_route(self, route):
        """Optimize a route by removing unnecessary waypoints"""
        if len(route) <= 2:
            return route
        
        # Try to remove waypoints while keeping the route valid
        i = 1
        while i < len(route) - 1:
            # Check if we can remove this waypoint
            prev_point = route[i-1]
            next_point = route[i+1]
            
            # Check if direct path between prev and next is valid
            valid_direct_path = True
            
            # Sample points along the direct path
            num_samples = 10
            for j in range(1, num_samples + 1):
                ratio = j / (num_samples + 1)
                sample_lat = prev_point[0] + (next_point[0] - prev_point[0]) * ratio
                sample_lon = prev_point[1] + (next_point[1] - prev_point[1]) * ratio
                
                if self.terrain.is_land(sample_lat, sample_lon):
                    valid_direct_path = False
                    break
            
            if valid_direct_path:
                # Remove the waypoint
                route.pop(i)
            else:
                # Keep the waypoint and move to the next one
                i += 1
        
        return route

class ShipRouting:
    """Main class for dynamic ship routing with land avoidance"""
    
    def __init__(self, config=None):
        if config is None:
            config = CONFIG
        
        self.config = config
        self.terrain = TerrainRecognition(config)
        self.ais = AISDataIntegration(config)
        self.pathfinding = PathfindingAlgorithms(self.terrain)
    
    def calculate_route(self, start, end, ship_type="container_medium"):
        """Calculate optimal route from start to end"""
        print(f"Calculating route from {start} to {end} for {ship_type}...")
        
        # Calculate direct distance
        direct_distance = self.pathfinding.haversine_distance(
            start[0], start[1], end[0], end[1]
        )
        print(f"Direct distance: {direct_distance:.1f} nautical miles")
        
        # Check if this is a high-risk route (likely crosses major land masses)
        is_high_risk = self.is_high_risk_route(start, end)
        
        if is_high_risk:
            print("Detected high-risk land crossing route. Using A* pathfinding...")
            route = self.pathfinding.a_star_pathfinding(start, end)
        else:
            print("Using direct route with waypoints...")
            route = self.pathfinding.create_direct_route_with_waypoints(start, end)
        
        # Optimize the route
        optimized_route = self.pathfinding.optimize_route(route)
        
        # Calculate actual distance
        actual_distance = 0
        for i in range(1, len(optimized_route)):
            wp1 = optimized_route[i-1]
            wp2 = optimized_route[i]
            segment_distance = self.pathfinding.haversine_distance(
                wp1[0], wp1[1], wp2[0], wp2[1]
            )
            actual_distance += segment_distance
        
        print(f"Optimized route distance: {actual_distance:.1f} nautical miles")
        print(f"Distance increase: {((actual_distance/direct_distance)-1)*100:.1f}%")
        
        # Get traffic density along the route
        traffic = self.ais.get_traffic_density(optimized_route)
        
        # Return complete route information
        return {
            "start": start,
            "end": end,
            "ship_type": ship_type,
            "waypoints": optimized_route,
            "distance": actual_distance,
            "direct_distance": direct_distance,
            "traffic_density": traffic,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def is_high_risk_route(self, start, end):
        """Check if route is high-risk (likely crosses major land masses)"""
        start_lat, start_lon = start
        end_lat, end_lon = end
        
        # Routes from India/South Asia to East Asia
        if ((start_lat > 5 and start_lat < 25 and start_lon > 70 and start_lon < 90) and 
            (end_lat > 20 and end_lat < 40 and end_lon > 110 and end_lon < 130)):
            return True
            
        # Routes within Indian subcontinent that cross peninsula
        if ((start_lat > 5 and start_lat < 25 and start_lon > 68 and start_lon < 90) and 
            (end_lat > 5 and end_lat < 25 and end_lon > 68 and end_lon < 90)):
            # Check if route crosses Indian peninsula
            if ((start_lon < 78 and end_lon > 82) or (start_lon > 82 and end_lon < 78)):
                return True
        
        # Mediterranean to Red Sea (Suez Canal)
        if ((start_lat > 30 and start_lat < 45 and start_lon > 0 and start_lon < 40) and
            (end_lat > 10 and end_lat < 30 and end_lon > 30 and end_lon < 60)):
            return True
            
        # Pacific routes crossing Central America
        if ((abs(start_lon - end_lon) > 40) and 
            (min(start_lat, end_lat) < 20) and (max(start_lat, end_lat) < 30)):
            return True
        
        # Simple check for any route that crosses a continent
        # Sample points along the direct route
        num_samples = 20
        land_points = 0
        
        for i in range(1, num_samples + 1):
            ratio = i / (num_samples + 1)
            sample_lat = start_lat + (end_lat - start_lat) * ratio
            sample_lon = start_lon + (end_lon - start_lon) * ratio
            
            if self.terrain.is_land(sample_lat, sample_lon):
                land_points += 1
                
                # If more than 30% of samples are on land, it's high risk
                if land_points > num_samples * 0.3:
                    return True
        
        return False
    
    def visualize_route(self, route_data, output_file=None):
        """Visualize calculated route on a map"""
        waypoints = route_data["waypoints"]
        start = route_data["start"]
        end = route_data["end"]
        
        # Extract latitude and longitude arrays
        lats = [wp[0] for wp in waypoints]
        lons = [wp[1] for wp in waypoints]
        
        plt.figure(figsize=(12, 8))
        
        # Define min/max bounds with padding
        min_lat = min(lats) - 2
        max_lat = max(lats) + 2
        min_lon = min(lons) - 2
        max_lon = max(lons) + 2
        
        lat_step = 0.5
        lon_step = 0.5
        
        # Create a grid of points to check for land/water
        grid_lats = np.arange(min_lat, max_lat, lat_step)
        grid_lons = np.arange(min_lon, max_lon, lon_step)
        
        # Create a land/water grid
        land_water_grid = np.zeros((len(grid_lats), len(grid_lons)))
        
        for i, lat in enumerate(grid_lats):
            for j, lon in enumerate(grid_lons):
                land_water_grid[i, j] = 0.0 if self.terrain.is_land(lat, lon) else 1.0
        
        # Plot the land/water grid
        plt.pcolormesh(grid_lons, grid_lats, land_water_grid, 
                       cmap=colors.ListedColormap(['lightgreen', 'lightblue']),
                       alpha=0.7)
        
        # Plot the route
        plt.plot(lons, lats, 'r-', linewidth=2, label='Ship Route')
        
        # Mark start and end points
        plt.plot(start[1], start[0], 'go', markersize=10, label='Start')
        plt.plot(end[1], end[0], 'bo', markersize=10, label='End')
        
        # Mark waypoints
        for i, (lat, lon) in enumerate(waypoints):
            if i != 0 and i != len(waypoints) - 1:  # Skip start and end
                plt.plot(lon, lat, 'yo', markersize=6)
                plt.text(lon, lat, str(i), fontsize=8)
        
        # Add title and labels
        plt.title(f'Ship Route: {start} to {end}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Add distance information
        distance_text = (
            f"Total Route Distance: {route_data['distance']:.1f} nm\n"
            f"Direct Distance: {route_data['direct_distance']:.1f} nm\n"
            f"Increase: {((route_data['distance']/route_data['direct_distance'])-1)*100:.1f}%"
        )
        plt.figtext(0.02, 0.02, distance_text, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        
        # Save if output filename is provided
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Route visualization saved to {output_file}")
        
        # Show plot
        plt.show()

def test_route(start, end, ship_type="container_medium", visualize=True):
    """Test the routing system with a specific route"""
    # Initialize the routing system
    router = ShipRouting()
    
    # Calculate route
    route = router.calculate_route(start, end, ship_type)
    
    # Save route to JSON file
    filename = f"route_{int(start[0])}_{int(start[1])}_to_{int(end[0])}_{int(end[1])}.json"
    with open(filename, 'w') as f:
        json.dump(route, f, indent=2)
    
    print(f"Route saved to {filename}")
    
    # Visualize if requested
    if visualize:
        router.visualize_route(route, f"route_{int(start[0])}_{int(start[1])}_to_{int(end[0])}_{int(end[1])}.png")
    
    return route

def main():
    """Main function to demonstrate the ship routing system"""
    print("===== Dynamic Ship Routing System =====")
    print("Calculating routes with land avoidance...")
    
    # Define test routes
    test_routes = [
        # Challenge route: Visakhapatnam to Shanghai (crosses Southeast Asia)
        {
            "name": "Visakhapatnam to Shanghai",
            "start": [17.6868, 83.2185],
            "end": [31.2304, 121.4737]
        },
        # Mediterranean to Red Sea (Suez Canal)
        {
            "name": "Barcelona to Dubai",
            "start": [41.3851, 2.1734],
            "end": [25.2048, 55.2708]
        },
        # Cross-Atlantic
        {
            "name": "New York to Lisbon",
            "start": [40.7128, -74.0060],
            "end": [38.7223, -9.1393]
        }
    ]
    
    # Calculate and visualize routes
    results = []
    for route in test_routes:
        print(f"\nCalculating route for {route['name']}...")
        result = test_route(route["start"], route["end"])
        results.append({
            "name": route["name"],
            "result": result
        })
    
    print("\n===== Results Summary =====")
    for result in results:
        print(f"{result['name']}:")
        print(f"  - Distance: {result['result']['distance']:.1f} nm")
        print(f"  - Direct distance: {result['result']['direct_distance']:.1f} nm")
        print(f"  - Increase: {((result['result']['distance']/result['result']['direct_distance'])-1)*100:.1f}%")
        print(f"  - Waypoints: {len(result['result']['waypoints'])}")
    
    print("\nDone.")

if __name__ == "__main__":
    main()
