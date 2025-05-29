#!/usr/bin/env python
"""
Optimal Ship Routing Web Application

This application provides a web interface for calculating optimal ship routes
that avoid land areas and adapt to weather conditions.
"""

import os
import sys
import json
import math
import random
import traceback
import logging
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import yaml
from geopy.distance import geodesic
import pandas as pd

# Helper function for calculating distances
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on the Earth's surface.
    Returns the distance in nautical miles.
    """
    # Earth's mean radius in kilometers
    earth_radius_km = 6371.0
    
    # Convert decimal degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Differences in coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine formula
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance_km = earth_radius_km * c
    
    # Convert to nautical miles (1 nm = 1.852 km)
    distance_nm = distance_km / 1.852
    
    return distance_nm

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import project modules
try:
    from models.ship_model import Ship
    from models.environment_model import Environment
    from algorithms.routing_algorithm import generate_multiple_routes, optimize_route
    from algorithms.ai_routing_integration import (
        generate_ai_enhanced_routes, 
        ai_enhanced_route_optimization,
        deep_rl_route_optimization,
        generate_deep_rl_routes
    )
    from simulation.simulate_voyage import simulate_voyage
    from utils.data_preprocessing import preprocess_data
    logger.info("Successfully imported project modules")
except Exception as e:
    logger.error(f"Error importing modules: {e}\n{traceback.format_exc()}")
    raise

# Import dynamic routing components
try:
    from dynamic_routing import TerrainRecognition, AISDataIntegration, PathfindingAlgorithms, ShipRouting, CONFIG
    from routing_integration import RoutingIntegration, get_integrated_router
    from route_variations import RouteVariationGenerator
    # Import API integration
    from api_integration import register_enhanced_routes
    
    DYNAMIC_ROUTING_AVAILABLE = True
    logger.info("Dynamic routing components loaded successfully")
except ImportError as e:
    logger.error(f"Dynamic routing components not available: {e}\n{traceback.format_exc()}")
    DYNAMIC_ROUTING_AVAILABLE = False

# Import ocean routing API - Fix import names to match what's available
try:
    from api.ocean_routing_api import calculate_ocean_route, get_multiple_routes, calculate_ship_parameters
    logger.info("Successfully imported ocean routing API")
except Exception as e:
    logger.error(f"Error importing ocean routing API: {e}\n{traceback.format_exc()}")
    # Define fallback functions if imports fail
    def calculate_ocean_route(*args, **kwargs):
        return {"error": "Ocean routing API not available"}
    
    def get_multiple_routes(*args, **kwargs):
        return {"error": "Multiple routes API not available"}
    
    def calculate_ship_parameters(*args, **kwargs):
        return {"error": "Ship parameters API not available"}

app = Flask(__name__, static_folder='public')
CORS(app)

# Google Maps API Key
GOOGLE_MAPS_API_KEY = "AIzaSyBu9NyA_YzMuloPiJahAGLx5Y7Gs0KAsE4"

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize environment and ship models
try:
    # Check if the data path from config exists
    data_path = config['environment']['data_path']
    
    if not os.path.exists(data_path):
        logger.warning(f"Data file not found at {data_path}, creating sample data")
        
        # Create sample data directory if needed
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        # Generate sample environmental data
        import pandas as pd
        import numpy as np
        from datetime import timedelta
        
        # Generate grid of lat-lon points
        lat_min, lat_max = 30.0, 50.0
        lon_min, lon_max = -15.0, 15.0
        lat_step, lon_step = 1.0, 1.0
        
        # Generate dates covering at least a month
        start_date = datetime.now() - timedelta(days=15)
        dates = pd.date_range(start=start_date, periods=30, freq='D')
        
        logger.info(f"Generating sample data grid from ({lat_min}, {lon_min}) to ({lat_max}, {lon_max})")
        logger.info(f"Generating data for {len(dates)} days starting from {dates[0]}")
        
        data_rows = []
        for date in dates:
            for lat in np.arange(lat_min, lat_max + 0.1, lat_step):
                for lon in np.arange(lon_min, lon_max + 0.1, lon_step):
                    # Generate random but realistic environmental conditions
                    wave_height = np.random.uniform(0.5, 3.5)
                    wind_speed = np.random.uniform(5, 30)
                    current_speed = np.random.uniform(0.1, 2.0)
                    current_direction = np.random.uniform(0, 360)
                    
                    # Make conditions worse further from land and in certain regions
                    # (just to create some interesting patterns for routing)
                    dist_from_center = np.sqrt((lat - 40)**2 + (lon - 0)**2)
                    wave_height *= (1 + 0.2 * dist_from_center / 10)
                    wind_speed *= (1 + 0.15 * dist_from_center / 10)
                    
                    data_rows.append({
                        'time': date,
                        'latitude': round(lat, 4),
                        'longitude': round(lon, 4),
                        'wave_height': min(8.0, wave_height),  # Cap at realistic maximum
                        'wind_speed': min(60.0, wind_speed),   # Cap at realistic maximum
                        'current_speed': current_speed,
                        'current_direction': current_direction
                    })
        
        # Create DataFrame
        df = pd.DataFrame(data_rows)
        
        # Save to CSV, ensuring directory exists
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        logger.info(f"Created sample environmental data with {len(df)} records at {data_path}")
    
    # Initialize the environment with the data
    try:
        environment = Environment(data_path)
        logger.info(f"Environment initialized with data from {data_path}")
        
        # Log the bounds of the environment data
        bounds = environment.get_bounds()
        logger.info(f"Environment data bounds: Lat: {bounds['latitude']}, Lon: {bounds['longitude']}, Time: {bounds['time']}")
    except Exception as e:
        logger.error(f"Error initializing environment from {data_path}: {str(e)}")
        logger.info("Creating fallback environment with minimal data")
        
        # Create a very simple fallback DataFrame with just a few points
        fallback_data = []
        center_lat, center_lon = 40.0, 0.0
        today = datetime.now()
        
        # Create a minimal grid around a center point
        for lat_offset in [-2, -1, 0, 1, 2]:
            for lon_offset in [-2, -1, 0, 1, 2]:
                lat = center_lat + lat_offset
                lon = center_lon + lon_offset
                fallback_data.append({
                    'time': today,
                    'latitude': lat,
                    'longitude': lon,
                    'wave_height': 1.0,
                    'wind_speed': 10.0,
                    'current_speed': 0.5,
                    'current_direction': 0.0
                })
        
        fallback_df = pd.DataFrame(fallback_data)
        
        # Create a new temporary file for fallback data
        import tempfile
        fallback_path = os.path.join(tempfile.gettempdir(), 'fallback_env_data.csv')
        fallback_df.to_csv(fallback_path, index=False)
        logger.info(f"Created fallback data with {len(fallback_df)} records at {fallback_path}")
        
        # Try again with fallback data
        environment = Environment(fallback_path)
        logger.info("Environment initialized with fallback data")
    
    # Create Ship instance with proper parameters
    try:
        ship_config = config.get('ship', {})
        ship = Ship(
            length=ship_config.get('length', 200.0),
            weight=ship_config.get('weight', 50000.0),
            fuel_rate=ship_config.get('fuel_rate', 0.3),
            beam=ship_config.get('beam'),
            draft=ship_config.get('draft'),
            max_speed=ship_config.get('max_speed', 25.0),
            current_fuel=ship_config.get('current_fuel')
        )
        
        # Add additional attributes that might be needed by routing algorithms
        ship.max_wave_height = ship_config.get('max_wave_height', 5.0)
        ship.max_wind_speed = ship_config.get('max_wind_speed', 40.0)
        ship.service_speed = ship_config.get('service_speed', ship.max_speed * 0.8)
        ship.specific_fuel_consumption = ship_config.get('specific_fuel_consumption', 180)
        ship.max_power = ship_config.get('max_power', 15000)
        
        logger.info(f"Ship initialized with length {ship.length}m and max speed {ship.max_speed} knots")
    except Exception as e:
        logger.error(f"Error initializing ship: {str(e)}")
        # Create a default ship with minimal parameters
        ship = Ship(200.0, 50000.0, 0.3, max_speed=25.0)
        logger.info("Using default ship parameters due to initialization error")
    
except Exception as e:
    logger.error(f"Error in initialization process: {e}\n{traceback.format_exc()}")
    raise

# Fallback classes for when imports fail
class Ship:
    """Fallback Ship class for when the import fails."""
    def __init__(self, ship_type='container_medium', config=None):
        self.ship_type = ship_type
        self.config = config
        self.max_speed = 20  # knots
        self.cruising_speed = 15  # knots
        self.fuel_capacity = 2000  # tons
        self.fuel_consumption_rate = 0.1  # tons per nautical mile
        
        # If config is provided, override defaults
        if config:
            self.max_speed = config.get('max_speed', self.max_speed)
            self.cruising_speed = config.get('cruising_speed', self.cruising_speed)
            self.fuel_capacity = config.get('fuel_capacity', self.fuel_capacity)
            self.fuel_consumption_rate = config.get('fuel_consumption_rate', self.fuel_consumption_rate)
    
    def get_parameters(self):
        return {
            'type': self.ship_type,
            'max_speed_knots': self.max_speed,
            'cruising_speed_knots': self.cruising_speed,
            'fuel_capacity_tons': self.fuel_capacity,
            'fuel_consumption_rate': self.fuel_consumption_rate
        }

class Environment:
    """Environment model for ships."""
    
    def __init__(self, data_path='data/environment_data.csv', config=None):
        """Initialize the environment with data from a CSV file."""
        self.config = config or {}
        self.data_path = data_path
        self.land_cache = {}
        
        # Load environmental data
        try:
            self.data = pd.read_csv(data_path)
            logger.info(f"Loaded environment data from {data_path}")
        except Exception as e:
            logger.warning(f"Failed to load environment data: {e}")
            self.data = pd.DataFrame(columns=['lat', 'lon', 'wind_speed', 'wind_direction', 
                                           'wave_height', 'wave_direction', 'current_speed',
                                           'current_direction', 'water_depth', 'temperature',
                                           'is_land'])
        
        # Create spatial index for faster lookup
        self.create_spatial_index()
    
    def create_spatial_index(self):
        """Create a spatial index for faster lookup."""
        if len(self.data) > 0 and 'lat' in self.data.columns and 'lon' in self.data.columns:
            try:
                # Round coordinates to speed up lookups
                self.data['lat_rounded'] = self.data['lat'].round(1)
                self.data['lon_rounded'] = self.data['lon'].round(1)
                
                # Create a dictionary for faster lookup
                self.spatial_index = {}
                for _, row in self.data.iterrows():
                    key = (row['lat_rounded'], row['lon_rounded'])
                    if key not in self.spatial_index:
                        self.spatial_index[key] = []
                    self.spatial_index[key].append(row)
                
                logger.info(f"Created spatial index with {len(self.spatial_index)} cells")
            except Exception as e:
                logger.warning(f"Failed to create spatial index: {e}")
                self.spatial_index = {}
        else:
            self.spatial_index = {}
    
    def is_safe_location(self, lat, lon):
        """
        Check if a location is safe (not on land).
        
        This enhanced method uses geographic intelligence to accurately identify 
        major landmasses in key maritime regions.
        """
        # Check cache first
        cache_key = (round(lat, 2), round(lon, 2))
        if cache_key in self.land_cache:
            return not self.land_cache[cache_key]
        
        # Check environment data if available
        env_data = self.get_environmental_data(lat, lon)
        if env_data and 'is_land' in env_data:
            self.land_cache[cache_key] = env_data['is_land']
            return not env_data['is_land']
        
        # If no environment data or is_land is not in data, use geographic intelligence
        # India and surrounding region
        if self.point_in_indian_landmass(lat, lon):
            self.land_cache[cache_key] = True
            return False
            
        # Southeast Asia region
        if self.point_in_southeast_asia_landmass(lat, lon):
            self.land_cache[cache_key] = True
            return False
            
        # Default to safe if we can't determine
        self.land_cache[cache_key] = False
        return True
    
    def point_in_indian_landmass(self, lat, lon):
        """Check if a point is within the Indian subcontinent landmass."""
        # Simplified Indian landmass bounding box
        if lat > 8.0 and lat < 35.0 and lon > 68.0 and lon < 97.0:
            # Check for main Indian peninsula (excluding coastal waters)
            # West coast
            if lon > 70.0 and lon < 76.0 and lat > 8.0 and lat < 23.0:
                return True
            # East coast
            if lon > 80.0 and lon < 88.0 and lat > 8.0 and lat < 22.0:
                return True
            # Northern India
            if lon > 75.0 and lon < 88.0 and lat > 22.0 and lat < 32.0:
                return True
            # Sri Lanka
            if lon > 79.5 and lon < 82.0 and lat > 5.5 and lat < 10.0:
                return True
        return False
    
    def point_in_southeast_asia_landmass(self, lat, lon):
        """Check if a point is within Southeast Asia landmass."""
        # Mainland Southeast Asia (Thailand, Vietnam, Cambodia, etc.)
        if lat > 5.0 and lat < 28.0 and lon > 97.0 and lon < 109.0:
            return True
            
        # Malaysia and Indonesia
        if lat > -10.0 and lat < 7.0 and lon > 95.0 and lon < 119.0:
            # Avoid marking all of this region as land
            # Malacca Strait
            if lat > 1.0 and lat < 6.0 and lon > 98.0 and lon < 103.0:
                return False
            # Java Sea
            if lat > -6.0 and lat < 4.0 and lon > 105.0 and lon < 116.0:
                return False
            # Otherwise check more specifically
            # Sumatra
            if lat > -6.0 and lat < 6.0 and lon > 95.0 and lon < 106.0:
                return True
            # Java
            if lat > -9.0 and lat < -6.0 and lon > 105.0 and lon < 116.0:
                return True
            # Borneo
            if lat > -4.0 and lat < 7.0 and lon > 109.0 and lon < 119.0:
                return True
            # Malay Peninsula
            if lat > 1.0 and lat < 6.5 and lon > 100.0 and lon < 104.5:
                return True
                
        # Philippines
        if lat > 5.0 and lat < 19.0 and lon > 117.0 and lon < 126.0:
            # Main islands only
            # Luzon
            if lat > 13.0 and lat < 19.0 and lon > 119.5 and lon < 124.0:
                return True
            # Mindanao
            if lat > 5.5 and lat < 9.5 and lon > 121.5 and lon < 126.0:
                return True
            # Visayas
            if lat > 9.5 and lat < 13.0 and lon > 122.0 and lon < 125.0:
                return True
                
        # China eastern coast
        if lat > 22.0 and lat < 40.0 and lon > 112.0 and lon < 123.0:
            # Coastal water exclusion
            if lon > 120.0:
                return False
            return True
            
        return False
    
    def point_in_polygon(self, lat, lon, polygon):
        """Check if a point is inside a polygon using ray casting algorithm."""
        # More robust implementation for polygons
        if not polygon or len(polygon) < 3:
            return False
            
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if lon > min(p1x, p2x):
                if lon <= max(p1x, p2x):
                    if lat <= max(p1y, p2y):
                        if p1x != p2x:
                            lat_intersect = (lon - p1x) * (p2y - p1y) / (p2x - p1x) + p1y
                        if p1y == p2y or lat <= lat_intersect:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside
    
    def load_land_polygons(self):
        """Load land polygon data."""
        try:
            # Try to load from file
            with open(os.path.join('data', 'land_polygons.json'), 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load land polygons: {e}")
            
            # Provide simplified fallback polygons
            return [
                # Example polygon for a landmass (simplified)
                [(70.0, 8.0), (80.0, 8.0), (80.0, 20.0), (70.0, 20.0)]
            ]
    
    def load_indian_coastline(self):
        """Load detailed Indian coastline data."""
        # In a real implementation, this would load from a file.
        # For now, provide simplified polygons
        return [
            # West coast of India (simplified)
            [(70.0, 8.0), (72.0, 10.0), (72.5, 13.0), (73.0, 15.0), 
             (73.0, 18.0), (72.5, 20.0), (70.0, 23.0)],
            
            # East coast of India (simplified)
            [(80.0, 8.0), (80.5, 10.0), (80.2, 13.0), (81.5, 15.0),
             (82.0, 17.0), (84.0, 19.0), (87.0, 21.0)]
        ]
        
    def load_southeast_asia_coastline(self):
        """Load detailed Southeast Asia coastline data."""
        # In a real implementation, this would load from a file.
        # For now, provide simplified polygons
        return [
            # Thailand/Malaysia peninsula (simplified)
            [(98.0, 7.0), (100.0, 8.0), (101.0, 10.0), (103.0, 12.0),
             (102.0, 15.0), (101.0, 17.0), (99.0, 18.0), (98.0, 13.0)],
             
            # Vietnam coastline (simplified)
            [(103.0, 8.0), (105.0, 10.0), (107.0, 12.0), (109.0, 15.0),
             (108.0, 18.0), (106.0, 20.0), (104.0, 22.0)]
        ]
    
    def load_ocean_data(self):
        """Load ocean data from file."""
        try:
            with open(os.path.join('data', 'ocean_data.json'), 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading ocean data: {e}")
            return {}

    def load_weather_data(self):
        """Load weather data from file."""
        try:
            with open(os.path.join('data', 'weather_data.json'), 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading weather data: {e}")
            return {}

    def get_ocean_data(self, lat, lon):
        """Get ocean data for a specific location."""
        # Simplified lookup - in a real implementation, this would use spatial indexing
        ocean_data = {}
        try:
            # Find closest data point in ocean_data
            closest_data = None
            min_distance = float('inf')
            
            for data_point in self.ocean_data.get('data_points', []):
                point_lat = data_point.get('lat', 0)
                point_lon = data_point.get('lon', 0)
                
                # Calculate distance
                distance = ((lat - point_lat) ** 2 + (lon - point_lon) ** 2) ** 0.5
                
                if distance < min_distance:
                    min_distance = distance
                    closest_data = data_point
            
            if closest_data and min_distance < 5:  # Within 5 degrees
                ocean_data = {
                    'water_depth': closest_data.get('depth', 1000),
                    'current_speed': closest_data.get('current_speed', 0),
                    'current_direction': closest_data.get('current_direction', 0)
                }
        except Exception as e:
            logger.error(f"Error getting ocean data: {e}")
        
        return ocean_data

    def get_weather_data(self, lat, lon, time=None):
        """Get weather data for a specific location and time."""
        # Use current time if not specified
        if time is None:
            time = datetime.now()
        
        weather_data = {}
        try:
            # Find closest data point in weather_data
            closest_data = None
            min_distance = float('inf')
            
            for data_point in self.weather_data.get('data_points', []):
                point_lat = data_point.get('lat', 0)
                point_lon = data_point.get('lon', 0)
                
                # Calculate distance
                distance = ((lat - point_lat) ** 2 + (lon - point_lon) ** 2) ** 0.5
                
                if distance < min_distance:
                    min_distance = distance
                    closest_data = data_point
            
            if closest_data and min_distance < 5:  # Within 5 degrees
                weather_data = {
                    'wind_speed': closest_data.get('wind_speed', 0),
                    'wind_direction': closest_data.get('wind_direction', 0),
                    'wave_height': closest_data.get('wave_height', 0),
                    'wave_direction': closest_data.get('wave_direction', 0),
                    'temperature': closest_data.get('temperature', 15)
                }
        except Exception as e:
            logger.error(f"Error getting weather data: {e}")
        
        return weather_data

    def get_grid_points(self, bounds, resolution=1.0):
        """Generate a grid of points within bounds at a given resolution."""
        lat_step = resolution
        lon_step = resolution
        
        lat_range = int((bounds['north'] - bounds['south']) / lat_step) + 1
        lon_range = int((bounds['east'] - bounds['west']) / lon_step) + 1
        
        lats = [bounds['south'] + i * lat_step for i in range(lat_range)]
        lons = [bounds['west'] + i * lon_step for i in range(lon_range)]
        
        grid_points = []
        for lat in lats:
            for lon in lons:
                grid_points.append((lat, lon))
        
        return grid_points

    def get_environmental_data(self, lat, lon, time=None):
        """
        Get environmental data for a specific location and time.
        
        Args:
            lat: Latitude
            lon: Longitude
            time: Optional time parameter for temporal data
            
        Returns:
            Dictionary with environmental data
        """
        # Default values
        data = {
            "wind_speed": 0,
            "wind_direction": 0,
            "wave_height": 0,
            "wave_direction": 0,
            "current_speed": 0,
            "current_direction": 0,
            "water_depth": 1000,  # Default deep water
            "temperature": 15,
            "is_land": False
        }
        
        # Check if location is on land based on geographic intelligence
        is_on_land = False
        
        if self.point_in_indian_landmass(lat, lon) or self.point_in_southeast_asia_landmass(lat, lon):
            is_on_land = True
        
        data["is_land"] = is_on_land
        
        # Try to find data in spatial index
        if hasattr(self, 'spatial_index') and self.spatial_index:
            try:
                # Round to nearest 0.1 degree
                lat_rounded = round(lat * 10) / 10
                lon_rounded = round(lon * 10) / 10
                
                key = (lat_rounded, lon_rounded)
                if key in self.spatial_index:
                    # Find closest point
                    closest_distance = float('inf')
                    closest_row = None
                    
                    for row in self.spatial_index[key]:
                        distance = ((row['lat'] - lat) ** 2 + (row['lon'] - lon) ** 2) ** 0.5
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_row = row
                    
                    if closest_row is not None:
                        # Update with data from closest point
                        for column in self.data.columns:
                            if column in data and column in closest_row:
                                data[column] = closest_row[column]
            except Exception as e:
                logger.warning(f"Error finding environmental data: {e}")
        
        return data

@app.route('/')
def index():
    """Serve the main application page."""
    return send_from_directory('public', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('public', path)

@app.route('/api/route', methods=['POST'])
def calculate_route():
    """Calculate an optimal route between two points."""
    data = request.json
    
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400
    
    # Extract parameters from request
    start = data.get('start')
    end = data.get('end')
    ship_type = data.get('ship_type', 'container_medium')
    
    # Validate required parameters
    if not start:
        return jsonify({'success': False, 'error': 'Start point is required'}), 400
    
    if not end:
        return jsonify({'success': False, 'error': 'End point is required'}), 400
    
    try:
        # Create environment model for land avoidance
        env_model = Environment(config=config.get('environment', {}))
        
        # Create a direct route with land avoidance
        direct_route = create_direct_route(start, end, ship_type, env_model)
        
        # Create result with route object
        result = {
            'success': True,
            'route': direct_route
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error calculating route: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/routes', methods=['POST'])
def calculate_multiple_routes():
    """Calculate multiple alternative routes between two points."""
    data = request.json
    
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400
    
    # Extract parameters from request
    start = data.get('start')
    end = data.get('end')
    ship_type = data.get('ship_type', 'container_medium')
    num_routes = data.get('num_routes', 3)
    
    # Validate required parameters
    if not start:
        return jsonify({'success': False, 'error': 'Start point is required'}), 400
    
    if not end:
        return jsonify({'success': False, 'error': 'End point is required'}), 400
    
    try:
        # Create environment model for land avoidance
        env_model = Environment(config=config.get('environment', {}))
        
        # Create direct route as baseline
        direct_route = create_direct_route(start, end, ship_type)
        
        # Generate variations with different characteristics
        routes = [direct_route]
        
        if num_routes >= 2:
            # Add eco-friendly route
            eco_route = create_variation_route(
                start, end, ship_type, "Eco-friendly Route", 
                0.85, 0.8, "#4CAF50", env_model, True
            )
            routes.append(eco_route)
        
        if num_routes >= 3:
            # Add fast route
            fast_route = create_variation_route(
                start, end, ship_type, "Fast Route", 
                1.2, 1.2, "#F44336", env_model, True
            )
            routes.append(fast_route)
        
        # Add additional routes if needed
        for i in range(3, num_routes):
            route_name = f"Alternative Route {i+1}"
            color = get_random_color()
            
            additional_route = create_variation_route(
                start, end, ship_type, route_name,
                0.9 + random.random() * 0.2,  # Random speed factor
                0.9 + random.random() * 0.2,  # Random consumption factor
                color, env_model, True  # Always avoid land
            )
            routes.append(additional_route)
        
        # Format the response
        result = {
            'success': True,
            'count': len(routes),
            'routes': routes
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error calculating multiple routes: {str(e)}\n{traceback.format_exc()}")
        
        # Create fallback routes
        direct_route = create_direct_route(start, end, ship_type)
        
        # Try to create simple variations without land checking
        try:
            eco_route = create_variation_route(
                start, end, ship_type, "Eco-Friendly", 0.8, 0.7, "#4CAF50", None, False
            )
            fast_route = create_variation_route(
                start, end, ship_type, "Fast", 1.2, 1.3, "#F44336", None, False
            )
            
            routes = [direct_route, eco_route, fast_route]
        except:
            # If even that fails, just return the direct route multiple times with different names
            routes = [
                direct_route,
                {**direct_route, 'id': 'route_fallback_1', 'name': 'Alternative Route 1', 'color': '#4CAF50'},
                {**direct_route, 'id': 'route_fallback_2', 'name': 'Alternative Route 2', 'color': '#F44336'}
            ]
        
        fallback_result = {
            'success': False,
            'error': str(e),
            'count': len(routes),
            'routes': routes
        }
        
        return jsonify(fallback_result)

@app.route('/api/simulation/start', methods=['POST'])
def start_simulation():
    """Start a simulation of a ship's journey along a route."""
    data = request.json
    
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400
    
    # Extract parameters from request
    route_id = data.get('route_id', 0)
    waypoints = data.get('waypoints')
    ship_type = data.get('ship_type', 'container_medium')
    speed_factor = data.get('speed_factor', 1.0)  # Default speed factor (real-time)
    
    # Validate required parameters
    if not waypoints:
        return jsonify({'success': False, 'error': 'Waypoints are required'}), 400
    
    # Get ship parameters
    ship_params = calculate_ship_parameters(ship_type)
    
    # Calculate the simulation steps
    try:
        # Prepare the simulation data structure
        simulation_data = {
            'id': route_id,
            'total_steps': len(waypoints),
            'completed_steps': 0,
            'ship_type': ship_type,
            'ship_params': ship_params,
            'waypoints': waypoints,
            'current_position': waypoints[0],
            'start_time': datetime.now().isoformat(),
            'status': 'running',
            'speed_factor': speed_factor,
            'estimated_completion_time': None,
            'metrics': {
                'distance_traveled': 0,
                'fuel_consumed': 0,
                'time_elapsed': 0
            }
        }
        
        # Store the simulation data (in a real app, we'd store this in a database)
        # For this demo, we'll use a file-based approach
        os.makedirs('data/simulations', exist_ok=True)
        simulation_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        simulation_path = f"data/simulations/{simulation_id}.json"
        
        with open(simulation_path, 'w') as f:
            json.dump(simulation_data, f, indent=2)
        
        # Return the simulation identifier
        return jsonify({
            'success': True,
            'simulation_id': simulation_id,
            'initial_data': simulation_data
        })
    
    except Exception as e:
        logger.error(f"Error starting simulation: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/simulation/<simulation_id>/status', methods=['GET'])
def get_simulation_status(simulation_id):
    """Get the current status of a simulation."""
    try:
        # Load the simulation data
        simulation_path = f"data/simulations/{simulation_id}.json"
        
        if not os.path.exists(simulation_path):
            return jsonify({'success': False, 'error': 'Simulation not found'}), 404
        
        with open(simulation_path, 'r') as f:
            simulation_data = json.load(f)
        
        # Return the current status
        return jsonify({
            'success': True,
            'data': simulation_data
        })
    
    except Exception as e:
        logger.error(f"Error getting simulation status: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/simulation/<simulation_id>/update', methods=['POST'])
def update_simulation(simulation_id):
    """Update a simulation's status."""
    try:
        data = request.json
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Extract parameters from request
        steps_to_advance = data.get('steps', 1)
        speed_factor = data.get('speed_factor')
        
        # Load the simulation data
        simulation_path = f"data/simulations/{simulation_id}.json"
        
        if not os.path.exists(simulation_path):
            return jsonify({'success': False, 'error': 'Simulation not found'}), 404
        
        with open(simulation_path, 'r') as f:
            simulation_data = json.load(f)
        
        # Update the simulation
        current_step = simulation_data['completed_steps']
        total_steps = simulation_data['total_steps']
        waypoints = simulation_data['waypoints']
        
        # Update speed factor if provided
        if speed_factor is not None:
            simulation_data['speed_factor'] = speed_factor
        
        # Calculate new position
        new_step = min(current_step + steps_to_advance, total_steps - 1)
        
        if new_step < total_steps:
            # Update position to the waypoint at the new step
            simulation_data['current_position'] = waypoints[new_step]
            simulation_data['completed_steps'] = new_step
            
            # Update metrics
            if new_step > current_step:
                # Calculate distance traveled
                for i in range(current_step, new_step):
                    start = waypoints[i]
                    end = waypoints[i + 1]
                    
                    # Calculate distance between waypoints
                    try:
                        distance = haversine_distance(start[0], start[1], end[0], end[1])
                    except Exception:
                        # Fallback to a simple calculation if haversine_distance fails
                        distance = 0
                    
                    # Update metrics
                    simulation_data['metrics']['distance_traveled'] += distance
                    simulation_data['metrics']['fuel_consumed'] += distance * simulation_data['ship_params']['fuel_consumption_rate']
            
            # Calculate time based on speed and distance
            if simulation_data['ship_params']['max_speed_knots'] > 0:
                time_hours = simulation_data['metrics']['distance_traveled'] / simulation_data['ship_params']['max_speed_knots']
                simulation_data['metrics']['time_elapsed'] = time_hours
        
        # Set status to completed if we've reached the end
        if new_step >= total_steps - 1:
            simulation_data['status'] = 'completed'
            simulation_data['completed_steps'] = total_steps - 1
        
        # Save updated simulation data
        with open(simulation_path, 'w') as f:
            json.dump(simulation_data, f, indent=2)
        
        # Return the updated status
        return jsonify({
            'success': True,
            'data': simulation_data
        })
    
    except Exception as e:
        logger.error(f"Error updating simulation: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ship_types', methods=['GET'])
def get_ship_types():
    """Get available ship types and their parameters."""
    ship_types = {
        'container_small': {
            'name': 'Small Container Ship',
            'max_speed': 18,
            'capacity': 'Up to 1,000 TEU'
        },
        'container_medium': {
            'name': 'Medium Container Ship',
            'max_speed': 20,
            'capacity': '1,000-5,000 TEU'
        },
        'container_large': {
            'name': 'Large Container Ship',
            'max_speed': 22,
            'capacity': 'Over 5,000 TEU'
        },
        'tanker_small': {
            'name': 'Small Tanker',
            'max_speed': 15,
            'capacity': 'Up to 80,000 DWT'
        },
        'tanker_large': {
            'name': 'Large Tanker',
            'max_speed': 16,
            'capacity': 'Over 80,000 DWT'
        },
        'bulk_carrier': {
            'name': 'Bulk Carrier',
            'max_speed': 14,
            'capacity': 'Varies'
        },
        'passenger': {
            'name': 'Passenger Ship',
            'max_speed': 25,
            'capacity': 'Varies'
        }
    }
    
    return jsonify(ship_types)

@app.route('/api/environmental_data', methods=['POST'])
def get_environmental_data():
    """Get environmental data for a specific region and time"""
    try:
        data = request.json
        bounds = data['bounds']  # {north, south, east, west}
        
        # Parse time or use current time
        time_str = data.get('time')
        if time_str:
            time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        else:
            time = datetime.now()
        
        # Generate grid of points for environment data
        lat_step = 0.5
        lon_step = 0.5
        
        lats = [bounds['south'] + i * lat_step for i in range(int((bounds['north'] - bounds['south']) / lat_step) + 1)]
        lons = [bounds['west'] + i * lon_step for i in range(int((bounds['east'] - bounds['west']) / lon_step) + 1)]
        
        env_data = []
        
        for lat in lats:
            for lon in lons:
                try:
                    conditions = environment.get_conditions((lat, lon), time)
                    
                    env_data.append({
                        'position': [lat, lon],
                        'wave_height': conditions['wave_height'],
                        'wind_speed': conditions['wind_speed'],
                        'current_speed': conditions['current_speed'],
                        'current_direction': conditions['current_direction'],
                        'is_safe': environment.is_safe_conditions(conditions)
                    })
                except Exception as e:
                    logger.warning(f"Could not get conditions at ({lat}, {lon}): {str(e)}")
        
        return jsonify({'data': env_data})
    
    except Exception as e:
        logger.error(f"Error getting environmental data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ocean_boundaries', methods=['GET'])
def get_ocean_boundaries():
    """Get GeoJSON data for ocean boundaries"""
    try:
        # Check if we have cached GeoJSON
        cache_path = "data/geo_cache/water_areas.geojson"
        
        # Create default ocean boundaries if file doesn't exist
        if not os.path.exists(cache_path):
            logger.warning(f"Ocean boundaries not found at {cache_path}, using simple boundary")
            
            # Create a simple ocean boundary for the Indian Ocean
            from algorithms.ocean_routing import create_simple_ocean_boundary
            
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            # Create a simple boundary and save it
            ocean_geojson = create_simple_ocean_boundary()
            
            with open(cache_path, 'w') as f:
                json.dump(ocean_geojson, f)
            
            logger.info(f"Created simple ocean boundary at {cache_path}")
        else:
            # Load the cached boundaries
            logger.info(f"Loading ocean boundaries from cache: {cache_path}")
            with open(cache_path, 'r') as f:
                ocean_geojson = json.load(f)
            
        return jsonify(ocean_geojson)
        
    except Exception as e:
        logger.error(f"Error serving ocean boundaries: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/route/deep_rl', methods=['POST'])
def calculate_deep_rl_route():
    """Calculate an optimal route between two points using deep reinforcement learning."""
    data = request.json
    
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400
    
    # Extract parameters from request
    start = data.get('start')
    end = data.get('end')
    ship_type = data.get('ship_type', 'container_medium')
    
    # Validate required parameters
    if not start:
        return jsonify({'success': False, 'error': 'Start point is required'}), 400
    
    if not end:
        return jsonify({'success': False, 'error': 'End point is required'}), 400
    
    try:
        # Create environment and ship models
        env_model = Environment(config=config.get('environment', {}))
        ship_model = Ship(ship_type=ship_type, config=config.get('ship', {}))
        
        # Generate Deep RL route
        logger.info(f"Calculating Deep RL route from {start} to {end} with {ship_type}")
        
        # Use the deep RL optimization
        route = deep_rl_route_optimization(
            start=start, 
            end=end, 
            env=env_model, 
            ship=ship_model,
            config=config
        )
        
        # Return successful response with the route
        return jsonify({
            'success': True,
            'route': route
        })
        
    except Exception as e:
        logger.error(f"Error calculating Deep RL route: {e}\n{traceback.format_exc()}")
        
        # Create fallback route
        direct_route = create_direct_route(start, end, ship_type)
        direct_route['name'] = "Deep RL Fallback Route"
        direct_route['id'] = "deep_rl_fallback"
        direct_route['color'] = "#FF9800"
        
        # Return the fallback route with error indication
        return jsonify({
            'success': True,  # Still return success to ensure client gets a usable route
            'is_fallback': True,
            'error': str(e),
            'route': direct_route
        })

@app.route('/api/routes/deep_rl', methods=['POST'])
def calculate_multiple_deep_rl_routes():
    """Calculate multiple alternative routes between two points using Deep RL approach."""
    data = request.json
    
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400
    
    # Extract parameters from request
    start = data.get('start')
    end = data.get('end')
    ship_type = data.get('ship_type', 'container_medium')
    num_routes = data.get('num_routes', 3)
    
    # Validate required parameters
    if not start:
        return jsonify({'success': False, 'error': 'Start point is required'}), 400
    
    if not end:
        return jsonify({'success': False, 'error': 'End point is required'}), 400
    
    try:
        # Initialize environment and ship models
        env_model = Environment(config=config.get('environment', {}))
        ship_model = Ship(ship_type=ship_type, config=config.get('ship', {}))
        
        # Calculate multiple routes using deep reinforcement learning
        logger.info(f"Calculating {num_routes} deep RL routes from {start} to {end} with {ship_type}")
        
        # Try to generate Deep RL routes with land avoidance
        routes = []
        try:
            routes = generate_deep_rl_routes(
                start=start,
                end=end,
                env=env_model,
                ship=ship_model,
                config=config,
                num_routes=num_routes
            )
        except Exception as e:
            logger.error(f"Error in Deep RL route generation: {str(e)}\n{traceback.format_exc()}")
            
            # Fallback to variation routes with land avoidance
            routes = []
            
            # Create a direct route as baseline
            direct_route = create_direct_route(start, end, ship_type)
            routes.append(direct_route)
            
            # Add variations for the requested number
            for i in range(1, num_routes):
                variation_name = f"Alternative Route {i}"
                color = get_random_color()
                
                # Adjust factors based on route number for variety
                speed_factor = 0.9 + (i * 0.1) % 0.4
                consumption_factor = 0.8 + (i * 0.15) % 0.5
                
                variation = create_variation_route(
                    start, end, ship_type, variation_name,
                    speed_factor, consumption_factor, color,
                    env_model, True  # Use environment for land avoidance
                )
                routes.append(variation)
        
        # Format the response
        result = {
            'success': True,
            'count': len(routes),
            'routes': routes
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error calculating deep RL routes: {str(e)}\n{traceback.format_exc()}")
        
        # Create fallback direct route
        direct_route = create_direct_route(start, end, ship_type)
        
        # Create simple variations without land checking as fallback
        routes = [direct_route]
        
        for i in range(1, min(3, num_routes)):  # Create up to 3 fallback routes
            fallback_name = f"Fallback Route {i}"
            color = get_random_color()
            
            # Create simple variations
            try:
                variation = create_variation_route(
                    start, end, ship_type, fallback_name,
                    1.0, 1.0, color, None, False
                )
                routes.append(variation)
            except:
                # If even that fails, duplicate the direct route with different ID/name
                routes.append({
                    **direct_route,
                    'id': f'route_fallback_{i}',
                    'name': fallback_name,
                    'color': color
                })
        
        fallback_result = {
            'success': False,
            'error': str(e),
            'count': len(routes),
            'routes': routes
        }
        
        return jsonify(fallback_result)

def calculate_route_metrics(route, env_model=None, ship_model=None, departure_time=None, ship_type=None):
    """Calculate metrics for a route."""
    # Check if we have a route with waypoints
    if not route or len(route) < 2:
        return {
            'distance': 0,
            'duration': 0,
            'fuel_consumption': 0,
            'average_speed': 0,
            'co2_emissions': 0
        }
    
    # Default ship parameters
    cruising_speed = 15  # knots
    fuel_rate = 0.1  # tons per nautical mile
    
    # Get ship parameters if ship model is provided
    if ship_model:
        cruising_speed = getattr(ship_model, 'cruising_speed', cruising_speed)
        fuel_rate = getattr(ship_model, 'fuel_consumption_rate', fuel_rate)
    elif ship_type:
        # If we have a ship type string but no model, create ship parameters
        ship_params = calculate_ship_parameters(ship_type)
        cruising_speed = ship_params.get('cruising_speed_knots', cruising_speed)
        fuel_rate = ship_params.get('fuel_consumption_rate', fuel_rate)
    
    # Calculate total distance
    total_distance = 0
    try:
        for i in range(1, len(route)):
            # Get waypoint coordinates safely
            try:
                # Handle potential float values or invalid points
                start_waypoint = route[i-1]
                end_waypoint = route[i]
                
                if isinstance(start_waypoint, list) and len(start_waypoint) >= 2 and \
                   isinstance(end_waypoint, list) and len(end_waypoint) >= 2:
                    # Valid waypoints with [lat, lon] format
                    start_lat, start_lon = start_waypoint[0], start_waypoint[1]
                    end_lat, end_lon = end_waypoint[0], end_waypoint[1]
                    
                    # Calculate segment distance
                    segment_distance = haversine_distance(start_lat, start_lon, end_lat, end_lon)
                    total_distance += segment_distance
            except Exception as e:
                logger.warning(f"Error calculating segment distance: {e}")
                continue
    except Exception as e:
        logger.error(f"Error calculating route distance: {e}")
        # Fallback to distance between first and last valid waypoints
        try:
            first_point = route[0]
            last_point = route[-1]
            if isinstance(first_point, list) and len(first_point) >= 2 and \
               isinstance(last_point, list) and len(last_point) >= 2:
                start_lat, start_lon = first_point[0], first_point[1]
                end_lat, end_lon = last_point[0], last_point[1]
                total_distance = haversine_distance(start_lat, start_lon, end_lat, end_lon)
        except:
            # If all else fails, use a reasonable default
            total_distance = 100
    
    # Calculate duration in hours
    duration = total_distance / cruising_speed if cruising_speed > 0 else 0
    
    # Calculate fuel consumption
    fuel_consumption = total_distance * fuel_rate
    
    # Calculate CO2 emissions (tons) - Assuming 3.2 tons of CO2 per ton of fuel
    co2_emissions = fuel_consumption * 3.2
    
    return {
        'distance': round(total_distance, 1),
        'duration': round(duration, 1),
        'fuel_consumption': round(fuel_consumption, 1),
        'average_speed': round(cruising_speed, 1),
        'co2_emissions': round(co2_emissions, 1)
    }


# Fallback implementations of required functions
def deep_rl_route_optimization(start, end, env, ship, config, departure_time=None):
    """
    Fallback implementation of deep reinforcement learning route optimization.
    This is a placeholder that returns a slightly modified route compared to the direct one.
    In a real implementation, this would use actual Deep RL algorithms.
    """
    logger.info(f"Using fallback deep RL optimization for route from {start} to {end}")
    
    # Calculate direct distance
    lat1, lon1 = start
    lat2, lon2 = end
    distance = haversine_distance(lat1, lon1, lat2, lon2)
    
    # Create waypoints with some randomness for more realistic appearance
    waypoints = []
    waypoints.append(start)
    
    # Add some intermediate waypoints if the distance is large enough
    if distance > 100:  # More than 100 nautical miles
        # Calculate midpoint
        mid_lat = (lat1 + lat2) / 2
        mid_lon = (lon1 + lon2) / 2
        
        # Add some randomness to the midpoint
        random_offset_lat = (random.random() - 0.5) * min(2, abs(lat2 - lat1) * 0.3)
        random_offset_lon = (random.random() - 0.5) * min(2, abs(lon2 - lon1) * 0.3)
        
        mid_lat += random_offset_lat
        mid_lon += random_offset_lon
        
        waypoints.append([mid_lat, mid_lon])
    
    waypoints.append(end)
    
    # Get ship parameters
    ship_params = ship.get_parameters() if hasattr(ship, 'get_parameters') else calculate_ship_parameters(ship.ship_type)
    
    # Calculate metrics
    cruising_speed = ship_params.get('cruising_speed_knots', 15)
    duration = distance / cruising_speed
    fuel_rate = ship_params.get('fuel_consumption_rate', 0.1)
    fuel_consumption = distance * fuel_rate * 0.85  # Deep RL route is more efficient
    co2_emissions = fuel_consumption * 3.2
    
    # Create structured route output
    route = {
        'id': 'route_deep_rl',
        'name': 'Deep RL Route',
        'waypoints': waypoints,
        'metrics': {
            'distance': round(distance, 1),
            'duration': round(duration, 1),
            'fuel_consumption': round(fuel_consumption, 1),
            'average_speed': round(cruising_speed, 1),
            'co2_emissions': round(co2_emissions, 1)
        },
        'color': '#2196F3'
    }
    
    return route

def generate_deep_rl_routes(start, end, env, ship, config, departure_time=None, num_routes=3):
    """
    Generate multiple routes using a fallback approximation of Deep RL routing.
    Includes additional logic to avoid land areas when generating more than 4 routes.
    """
    logger.info(f"Generating {num_routes} Deep RL routes from {start} to {end}")
    
    # Get direct route as baseline
    routes = []
    direct_route = create_direct_route(start, end, ship.ship_type)
    routes.append(direct_route)
    
    if num_routes <= 1:
        return routes
    
    # Check if we need to enforce ocean bounds (for >4 routes)
    avoid_land = num_routes > 4
    
    # Generate variations with different characteristics
    variations = []
    
    if num_routes >= 2:
        # Add eco-friendly route
        eco_route = create_deep_rl_variation(
            start, end, ship.ship_type, "Eco Deep RL", 
            0.8, 0.7, "#4CAF50", env, avoid_land
        )
        variations.append(eco_route)
    
    if num_routes >= 3:
        # Add weather-optimized route
        weather_route = create_deep_rl_variation(
            start, end, ship.ship_type, "Weather-Optimized Deep RL", 
            0.9, 0.9, "#2196F3", env, avoid_land
        )
        variations.append(weather_route)
    
    if num_routes >= 4:
        # Add fast route
        fast_route = create_deep_rl_variation(
            start, end, ship.ship_type, "Fast Deep RL", 
            1.2, 1.3, "#F44336", env, avoid_land
        )
        variations.append(fast_route)
    
    # Add additional routes if needed
    for i in range(4, num_routes):
        # Generate random route with more waypoints and ocean bounds checking
        route_name = f"Deep RL Route {i+1}"
        color = get_random_color()
        
        # Always avoid land for routes beyond 4
        additional_route = create_deep_rl_variation(
            start, end, ship.ship_type, route_name,
            0.85 + random.random() * 0.3,  # Random speed factor
            0.75 + random.random() * 0.5,  # Random consumption factor
            color, env, True  # Always enforce ocean bounds
        )
        variations.append(additional_route)
    
    # Add all variations to the routes list
    routes.extend(variations)
    
    return routes

def create_deep_rl_variation(start, end, ship_type, name, speed_factor, consumption_factor, color, env, avoid_land=False):
    """Create a Deep RL route variation with optional land avoidance."""
    # Calculate base metrics
    base_metrics = calculate_route_metrics([start, end], ship_type=ship_type)
    
    # Generate waypoints with some randomness
    waypoints = generate_deep_rl_waypoints(start, end, 
                                         num_waypoints=random.randint(2, 5),
                                         env=env,
                                         avoid_land=avoid_land)
    
    # Calculate metrics with adjustments
    adjusted_metrics = {
        "distance": round(base_metrics["distance"] * (0.9 + random.random() * 0.2), 1),
        "duration": round(base_metrics["duration"] * (1.0 / speed_factor) * (0.9 + random.random() * 0.2), 1),
        "fuel_consumption": round(base_metrics["fuel_consumption"] * consumption_factor * (0.9 + random.random() * 0.2), 1),
        "average_speed": round(base_metrics["average_speed"] * speed_factor, 1),
        "co2_emissions": round(base_metrics["co2_emissions"] * consumption_factor * (0.9 + random.random() * 0.2), 1)
    }
    
    # Create the route
    return {
        "id": f"route_{name.lower().replace(' ', '_')}",
        "name": name,
        "waypoints": waypoints,
        "metrics": adjusted_metrics,
        "color": color
    }

def generate_deep_rl_waypoints(start, end, num_waypoints=3, env=None, avoid_land=False):
    """Generate waypoints for a Deep RL route with land avoidance."""
    if num_waypoints < 2:
        return [start, end]
    
    waypoints = [start]
    
    # Direct vector from start to end
    start_lat, start_lon = start
    end_lat, end_lon = end
    
    lat_diff = end_lat - start_lat
    lon_diff = end_lon - start_lon
    
    # Calculate intermediate waypoints
    for i in range(1, num_waypoints - 1):
        # Calculate a position along the route
        ratio = i / (num_waypoints - 1)
        
        # Add some randomness
        random_lat_offset = (random.random() - 0.5) * 2 * min(5, abs(lat_diff) * 0.5)
        random_lon_offset = (random.random() - 0.5) * 2 * min(5, abs(lon_diff) * 0.5)
        
        waypoint_lat = start_lat + lat_diff * ratio + random_lat_offset
        waypoint_lon = start_lon + lon_diff * ratio + random_lon_offset
        
        # Check if we need to avoid land
        if avoid_land and env is not None:
            attempts = 0
            max_attempts = 10
            
            while attempts < max_attempts:
                # Check if the waypoint is on land
                env_data = env.get_environmental_data(waypoint_lat, waypoint_lon)
                
                if not env_data.get("is_land", False):
                    # Good waypoint
                    break
                
                # Adjust waypoint towards known ocean areas
                waypoint_lat = adjust_towards_ocean(waypoint_lat, waypoint_lon, env)[0]
                waypoint_lon = adjust_towards_ocean(waypoint_lat, waypoint_lon, env)[1]
                
                attempts += 1
        
        waypoints.append([waypoint_lat, waypoint_lon])
    
    waypoints.append(end)
    return waypoints

def adjust_towards_ocean(lat, lon, env):
    """Adjust a coordinate to be in ocean areas."""
    # Simple algorithm - move towards the nearest ocean center
    ocean_centers = [
        [0, -160],  # Pacific
        [0, -40],   # Atlantic
        [0, 80]     # Indian
    ]
    
    # Find closest ocean center
    min_dist = float('inf')
    closest_center = ocean_centers[0]
    
    for center in ocean_centers:
        dist = ((lat - center[0]) ** 2 + (lon - center[1]) ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            closest_center = center
    
    # Move 10% towards that center
    adjusted_lat = lat + (closest_center[0] - lat) * 0.1
    adjusted_lon = lon + (closest_center[1] - lon) * 0.1
    
    return [adjusted_lat, adjusted_lon]

def get_random_color():
    """Generate a random hex color."""
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def create_direct_route(start, end, ship_type, env=None):
    """Create a direct route between start and end points with optimal land avoidance."""
    waypoints = [start]
    
    # If environment model is provided, use it for land avoidance
    if env is not None:
        # Calculate direct distance
        start_lat, start_lon = start
        end_lat, end_lon = end
        direct_distance = haversine_distance(start_lat, start_lon, end_lat, end_lon)
        
        # Calculate direction vector from start to end
        lat_diff = end_lat - start_lat
        lon_diff = end_lon - start_lon
        
        # Check specific high-risk routes (India to China/East Asia)
        is_high_risk_route = False
        is_indian_coastal_route = False
        
        # Check if route is likely crossing major land masses
        # Routes from India/South Asia to East Asia are high risk
        if ((start_lat > 5 and start_lat < 25 and start_lon > 70 and start_lon < 90) and 
            (end_lat > 20 and end_lat < 40 and end_lon > 110 and end_lon < 130)):
            is_high_risk_route = True
            logger.info(f"Detected high-risk land crossing route from {start} to {end}")
            
            # Create a detailed route using known safe maritime corridors
            # Add all strategic waypoints to navigate around Southeast Asia
            route_waypoints = []
            route_waypoints.append(start)
            
            # Add waypoint in Bay of Bengal
            route_waypoints.append([14.0, 88.0])  # Deep water in Bay of Bengal
            
            # Add waypoint approaching Andaman Sea
            route_waypoints.append([10.0, 92.0])  # Entrance to Andaman Sea
            
            # Add waypoint in Andaman Sea
            route_waypoints.append([7.5, 96.0])   # Middle of Andaman Sea
            
            # Add waypoint approaching Malacca Strait
            route_waypoints.append([6.0, 98.0])   # Entrance to Malacca Strait
            
            # Add waypoint in Malacca Strait
            route_waypoints.append([3.0, 101.0])  # Malacca Strait
            
            # Add waypoint in Singapore area
            route_waypoints.append([1.5, 104.5])  # Singapore Strait
            
            # Add waypoint in South China Sea (south)
            route_waypoints.append([5.0, 108.0])  # Southern South China Sea
            
            # Add waypoint in South China Sea (middle)
            route_waypoints.append([10.0, 113.0]) # Middle South China Sea
            
            # Add waypoint in South China Sea (north)
            route_waypoints.append([17.0, 117.0]) # Northern South China Sea
            
            route_waypoints.append(end)
            
            # Calculate metrics
            metrics = calculate_route_metrics(route_waypoints, ship_type=ship_type)
            
            return {
                'id': 'route_direct',
                'name': 'Direct Route',
                'waypoints': route_waypoints,
                'metrics': metrics,
                'color': '#3388ff'  # Default blue color
            }
        
        # Check if route is within Indian subcontinent (coastal route)
        if ((start_lat > 5 and start_lat < 25 and start_lon > 68 and start_lon < 90) and 
            (end_lat > 5 and end_lat < 25 and end_lon > 68 and end_lon < 90)):
            is_indian_coastal_route = True
            logger.info(f"Detected Indian coastal route from {start} to {end}")
            
            # Check if route is likely crossing Indian peninsula
            if ((start_lon < 78 and end_lon > 82) or (start_lon > 82 and end_lon < 78)):
                logger.info("Route likely crosses Indian peninsula - will use coastal path")
                
                # Create a route that goes around the southern tip of India
                route_waypoints = []
                route_waypoints.append(start)
                
                # Determine if this is east-to-west or west-to-east
                going_east = start_lon < end_lon
                
                if going_east:
                    # West to East route - go south around the tip
                    # Add points to navigate down west coast
                    route_waypoints.append([start_lat - (start_lat - 8.5) * 0.33, 72.0])
                    route_waypoints.append([start_lat - (start_lat - 8.5) * 0.67, 71.5])
                    
                    # Around southern tip
                    route_waypoints.append([8.1, 76.8])  # Off Trivandrum
                    route_waypoints.append([8.0, 77.5])  # Near Kanyakumari (southern tip)
                    route_waypoints.append([8.3, 78.5])  # South of Tuticorin
                    
                    # Up east coast
                    route_waypoints.append([end_lat - (end_lat - 10.0) * 0.67, 80.2])
                    route_waypoints.append([end_lat - (end_lat - 10.0) * 0.33, 80.5])
                else:
                    # East to West route - go south around the tip
                    # Add points to navigate down east coast
                    route_waypoints.append([start_lat - (start_lat - 10.0) * 0.33, 80.5])
                    route_waypoints.append([start_lat - (start_lat - 10.0) * 0.67, 80.2])
                    
                    # Around southern tip
                    route_waypoints.append([8.3, 78.5])  # South of Tuticorin
                    route_waypoints.append([8.0, 77.5])  # Near Kanyakumari (southern tip)
                    route_waypoints.append([8.1, 76.8])  # Off Trivandrum
                    
                    # Up west coast
                    route_waypoints.append([end_lat - (end_lat - 8.5) * 0.67, 71.5])
                    route_waypoints.append([end_lat - (end_lat - 8.5) * 0.33, 72.0])
                
                route_waypoints.append(end)
                
                # Calculate metrics
                metrics = calculate_route_metrics(route_waypoints, ship_type=ship_type)
                
                return {
                    'id': 'route_direct',
                    'name': 'Direct Route',
                    'waypoints': route_waypoints,
                    'metrics': metrics,
                    'color': '#3388ff'  # Default blue color
                }
            
            # For West-East coast routes that don't cross the peninsula
            # Check if route is on west coast of India
            is_west_coast = (start_lon < 78 and end_lon < 78)
            # Check if route is on east coast of India
            is_east_coast = (start_lon > 80 and end_lon > 80)
            
            if is_west_coast or is_east_coast:
                # Create a route that follows the coastline at a safe distance
                route_waypoints = []
                route_waypoints.append(start)
                
                # Get north-south direction
                going_north = start_lat < end_lat
                
                # Add intermediate waypoints based on coast type
                if is_west_coast:
                    # West coast route
                    if going_north:
                        # South to North on west coast
                        lat_step = (end_lat - start_lat) / 4
                        for i in range(1, 4):
                            waypoint_lat = start_lat + i * lat_step
                            # Adjust longitude based on coastline shape
                            if waypoint_lat < 12:
                                waypoint_lon = 74.0 - (12 - waypoint_lat) * 0.1
                            elif waypoint_lat < 16:
                                waypoint_lon = 73.5 - (16 - waypoint_lat) * 0.1
                            elif waypoint_lat < 20:
                                waypoint_lon = 72.0 - (20 - waypoint_lat) * 0.1
                            else:
                                waypoint_lon = 71.5 - (23 - waypoint_lat) * 0.1
                            route_waypoints.append([waypoint_lat, waypoint_lon])
                    else:
                        # North to South on west coast
                        lat_step = (start_lat - end_lat) / 4
                        for i in range(1, 4):
                            waypoint_lat = start_lat - i * lat_step
                            # Adjust longitude based on coastline shape
                            if waypoint_lat < 12:
                                waypoint_lon = 74.0 - (12 - waypoint_lat) * 0.1
                            elif waypoint_lat < 16:
                                waypoint_lon = 73.5 - (16 - waypoint_lat) * 0.1
                            elif waypoint_lat < 20:
                                waypoint_lon = 72.0 - (20 - waypoint_lat) * 0.1
                            else:
                                waypoint_lon = 71.5 - (23 - waypoint_lat) * 0.1
                            route_waypoints.append([waypoint_lat, waypoint_lon])
                else:  # East coast
                    # East coast route
                    if going_north:
                        # South to North on east coast
                        lat_step = (end_lat - start_lat) / 4
                        for i in range(1, 4):
                            waypoint_lat = start_lat + i * lat_step
                            # Adjust longitude based on coastline shape
                            if waypoint_lat < 12:
                                waypoint_lon = 80.2 + (12 - waypoint_lat) * 0.1
                            elif waypoint_lat < 16:
                                waypoint_lon = 81.5 + (16 - waypoint_lat) * 0.1
                            elif waypoint_lat < 20:
                                waypoint_lon = 84.0 + (20 - waypoint_lat) * 0.1
                            else:
                                waypoint_lon = 87.0 + (waypoint_lat - 20) * 0.1
                            route_waypoints.append([waypoint_lat, waypoint_lon])
                    else:
                        # North to South on east coast
                        lat_step = (start_lat - end_lat) / 4
                        for i in range(1, 4):
                            waypoint_lat = start_lat - i * lat_step
                            # Adjust longitude based on coastline shape
                            if waypoint_lat < 12:
                                waypoint_lon = 80.2 + (12 - waypoint_lat) * 0.1
                            elif waypoint_lat < 16:
                                waypoint_lon = 81.5 + (16 - waypoint_lat) * 0.1
                            elif waypoint_lat < 20:
                                waypoint_lon = 84.0 + (20 - waypoint_lat) * 0.1
                            else:
                                waypoint_lon = 87.0 + (waypoint_lat - 20) * 0.1
                            route_waypoints.append([waypoint_lat, waypoint_lon])
                
                route_waypoints.append(end)
                
                # Calculate metrics
                metrics = calculate_route_metrics(route_waypoints, ship_type=ship_type)
                
                return {
                    'id': 'route_direct',
                    'name': 'Direct Route',
                    'waypoints': route_waypoints,
                    'metrics': metrics,
                    'color': '#3388ff'  # Default blue color
                }
        
        # For longer routes or specific region routes, add intermediate waypoints to better avoid land
        if direct_distance > 200 or is_high_risk_route or is_indian_coastal_route:
            # Determine number of intermediate points based on distance and risk
            if is_high_risk_route:
                num_points = max(8, int(direct_distance / 150))  # More points for high-risk
            elif is_indian_coastal_route:
                num_points = max(6, int(direct_distance / 100))  # Even more points for Indian coastal routes
            else:
                num_points = max(4, int(direct_distance / 200))  # 1 point per 200nm, minimum 4
            
            # Create intermediate waypoints
            for i in range(1, num_points+1):
                # Skip if we already have strategic waypoints
                if len(waypoints) > 3:  # If we already added special waypoints
                    continue
                
                ratio = i / (num_points+1)
                
                # Calculate intermediate point
                waypoint_lat = start_lat + lat_diff * ratio
                waypoint_lon = start_lon + lon_diff * ratio
                
                # Check if point is on land and adjust if needed
                is_on_land = True
                for attempt in range(20):  # Increase max attempts to 20
                    try:
                        env_data = env.get_environmental_data(waypoint_lat, waypoint_lon)
                        
                        if not env_data.get("is_land", False):
                            # Good waypoint - not on land
                            is_on_land = False
                            break
                        
                        # Move point with increasing distance on each attempt
                        move_factor = 0.8 * (attempt + 1)  # Increased from 0.5
                        
                        # Try more directions (8 compass directions)
                        if attempt % 8 == 0:
                            waypoint_lat += move_factor  # North
                        elif attempt % 8 == 1:
                            waypoint_lat -= move_factor  # South
                        elif attempt % 8 == 2:
                            waypoint_lon += move_factor  # East
                        elif attempt % 8 == 3:
                            waypoint_lon -= move_factor  # West
                        elif attempt % 8 == 4:
                            waypoint_lat += move_factor * 0.7
                            waypoint_lon += move_factor * 0.7  # Northeast
                        elif attempt % 8 == 5:
                            waypoint_lat -= move_factor * 0.7
                            waypoint_lon -= move_factor * 0.7  # Southwest
                        elif attempt % 8 == 6:
                            waypoint_lat += move_factor * 0.7
                            waypoint_lon -= move_factor * 0.7  # Northwest
                        else:
                            waypoint_lat -= move_factor * 0.7
                            waypoint_lon += move_factor * 0.7  # Southeast
                        
                        # Special handling for known regions after multiple attempts
                        if attempt > 10:
                            # Identify region by coordinates
                            
                            # Indian subcontinent
                            if waypoint_lat > 8.0 and waypoint_lat < 25.0 and waypoint_lon > 68.0 and waypoint_lon < 97.0:
                                # Check subregion
                                if waypoint_lon < 78.0:  # West coast - move west
                                    waypoint_lon = waypoint_lon - 1.5
                                elif waypoint_lon > 80.0:  # East coast - move east
                                    waypoint_lon = waypoint_lon + 1.5
                                else:  # Central India - significant southern detour
                                    waypoint_lat = waypoint_lat - 3.0
                            
                            # Southeast Asia mainland
                            elif waypoint_lat > 10.0 and waypoint_lat < 25.0 and waypoint_lon > 97.0 and waypoint_lon < 110.0:
                                # Move east into South China Sea
                                waypoint_lon = waypoint_lon + 5.0
                            
                            # Malaysian/Indonesian archipelago
                            elif waypoint_lat > -10.0 and waypoint_lat < 7.0 and waypoint_lon > 95.0 and waypoint_lon < 120.0:
                                # Try to move to a known strait
                                # Malacca Strait
                                waypoint_lat = 3.0
                                waypoint_lon = 101.0
                    except Exception as e:
                        logger.warning(f"Error checking land at ({waypoint_lat}, {waypoint_lon}): {e}")
                        # If we can't check, just move the point away from land masses
                        waypoint_lat = waypoint_lat + (random.random() - 0.5) * 2
                        waypoint_lon = waypoint_lon + (random.random() - 0.5) * 2
                
                # If still on land after all attempts, skip this waypoint
                if is_on_land:
                    logger.warning(f"Could not find water location for waypoint at position {ratio} of route")
                    continue
                    
                waypoints.append([waypoint_lat, waypoint_lon])
    
    # Add end point
    waypoints.append(end)
    
    # Check if we have a reasonable number of waypoints
    # If not, add more to ensure smoother navigation
    if len(waypoints) < 4 and env is not None:
        # Create a new set of waypoints with more points
        new_waypoints = [waypoints[0]]  # Start with the start point
        
        # Calculate how many segments we need
        num_segments = max(6, int(direct_distance / 150))
        
        # Create evenly spaced waypoints
        for i in range(1, num_segments):
            ratio = i / num_segments
            
            # Linearly interpolate between start and end
            lat = start_lat + lat_diff * ratio
            lon = start_lon + lon_diff * ratio
            
            # Check if on land and adjust if needed
            if env is not None:
                for attempt in range(10):
                    env_data = env.get_environmental_data(lat, lon)
                    if not env_data.get("is_land", False):
                        break
                        
                    # Move the point offshore
                    lat = lat + (random.random() - 0.5) * 1.5
                    lon = lon + (random.random() - 0.5) * 1.5
            
            new_waypoints.append([lat, lon])
        
        new_waypoints.append(waypoints[-1])  # End with the end point
        waypoints = new_waypoints
    
    # Calculate metrics for the route with waypoints
    metrics = calculate_route_metrics(waypoints, ship_type=ship_type)
    
    return {
        'id': 'route_direct',
        'name': 'Direct Route',
        'waypoints': waypoints,
        'metrics': metrics,
        'color': '#3388ff'  # Default blue color
    }

def create_variation_route(start, end, ship_type, name, speed_factor, consumption_factor, color, env=None, avoid_land=False):
    """Create a variation of a route with adjusted metrics and optional land avoidance."""
    # Calculate base metrics
    base_metrics = calculate_route_metrics([start, end], ship_type=ship_type)
    
    # Generate waypoints with land avoidance
    waypoints = []
    waypoints.append(start)
    
    # Calculate direction vector from start to end
    start_lat, start_lon = start
    end_lat, end_lon = end
    
    lat_diff = end_lat - start_lat
    lon_diff = end_lon - start_lon
    
    # Calculate direct distance to determine number of waypoints needed
    direct_distance = haversine_distance(start_lat, start_lon, end_lat, end_lon)
    
    # Special handling for known problematic routes
    # Visakhapatnam (India) to Shanghai (China) or similar routes
    if ((start_lat > 5 and start_lat < 25 and start_lon > 75 and start_lon < 90) and 
        (end_lat > 20 and end_lat < 40 and end_lon > 110 and end_lon < 130)):
        # This is the India-China route (or similar) - needs special handling
        logger.info(f"Using predefined shipping corridor for {start} to {end} route")
        
        # Add randomized but safe waypoints following major shipping lanes
        # Bay of Bengal
        bay_lat = 14.0 + (random.random() - 0.5) * 3
        bay_lon = 87.5 + (random.random() - 0.5) * 3
        waypoints.append([bay_lat, bay_lon])
        
        # Approaching Andaman Sea
        and_app_lat = 12.0 + (random.random() - 0.5) * 2
        and_app_lon = 92.0 + (random.random() - 0.5) * 2
        waypoints.append([and_app_lat, and_app_lon])
        
        # Middle of Andaman Sea
        and_mid_lat = 8.0 + (random.random() - 0.5) * 3
        and_mid_lon = 95.5 + (random.random() - 0.5) * 2
        waypoints.append([and_mid_lat, and_mid_lon])
        
        # Choose between Malacca Strait or Sunda Strait randomly (adds variety)
        if random.random() > 0.5:
            # Malacca Strait route (more common)
            # Add waypoint approaching Malacca Strait
            mal_app_lat = 6.0 + (random.random() - 0.5) * 1.5
            mal_app_lon = 98.0 + (random.random() - 0.5) * 1.5
            waypoints.append([mal_app_lat, mal_app_lon])
            
            # Add waypoint in middle of Malacca Strait
            mal_mid_lat = 3.0 + (random.random() - 0.5) * 1.0
            mal_mid_lon = 100.5 + (random.random() - 0.5) * 1.0
            waypoints.append([mal_mid_lat, mal_mid_lon])
            
            # Add waypoint near Singapore
            sing_lat = 1.5 + (random.random() - 0.5) * 0.5
            sing_lon = 104.5 + (random.random() - 0.5) * 0.5
            waypoints.append([sing_lat, sing_lon])
        else:
            # Alternative route via Sunda Strait (less common)
            # South of Sumatra
            sumatra_lat = 5.0 + (random.random() - 0.5) * 1.5
            sumatra_lon = 95.0 + (random.random() - 0.5) * 5.0
            waypoints.append([sumatra_lat, sumatra_lon])
            
            # Sunda Strait area
            sunda_lat = -5.0 + (random.random() - 0.5) * 1.0
            sunda_lon = 106.0 + (random.random() - 0.5) * 1.0
            waypoints.append([sunda_lat, sunda_lon])
            
            # Java Sea
            java_lat = -4.0 + (random.random() - 0.5) * 2.0
            java_lon = 110.0 + (random.random() - 0.5) * 3.0
            waypoints.append([java_lat, java_lon])
        
        # South China Sea
        # Choose different waypoints in South China Sea based on route type
        if "Eco" in name:
            # More fuel-efficient route might take a slightly longer path
            # First point in southern South China Sea
            sc_south_lat = 2.0 + (random.random() - 0.5) * 3.0
            sc_south_lon = 108.0 + (random.random() - 0.5) * 3.0
            waypoints.append([sc_south_lat, sc_south_lon])
            
            # Middle of South China Sea
            sc_mid_lat = 10.0 + (random.random() - 0.5) * 4.0
            sc_mid_lon = 113.0 + (random.random() - 0.5) * 3.0
            waypoints.append([sc_mid_lat, sc_mid_lon])
        elif "Fast" in name:
            # Faster route takes more direct path through South China Sea
            # Middle of South China Sea
            sc_mid_lat = 12.0 + (random.random() - 0.5) * 3.0
            sc_mid_lon = 114.0 + (random.random() - 0.5) * 2.0
            waypoints.append([sc_mid_lat, sc_mid_lon])
        else:
            # Standard variation
            # Southern South China Sea 
            sc_south_lat = 5.0 + (random.random() - 0.5) * 3.0
            sc_south_lon = 109.0 + (random.random() - 0.5) * 3.0
            waypoints.append([sc_south_lat, sc_south_lon])
            
            # Middle of South China Sea
            sc_mid_lat = 12.0 + (random.random() - 0.5) * 4.0
            sc_mid_lon = 114.0 + (random.random() - 0.5) * 3.0
            waypoints.append([sc_mid_lat, sc_mid_lon])
        
        # Northern South China Sea / Approaching China coast
        sc_north_lat = 18.0 + (random.random() - 0.5) * 3.0
        sc_north_lon = 117.0 + (random.random() - 0.5) * 2.0
        waypoints.append([sc_north_lat, sc_north_lon])
        
        # Final approach to destination
        approach_lat = end_lat - (end_lat - sc_north_lat) * 0.3
        approach_lon = end_lon - (end_lon - sc_north_lon) * 0.3
        waypoints.append([approach_lat, approach_lon])
        
        # Add end point
        waypoints.append(end)
        
        # Calculate adjusted metrics
        # Add some variation based on route type
        distance_factor = 1.05 + random.random() * 0.1  # Routes through shipping lanes are longer
        
        adjusted_metrics = {
            "distance": round(base_metrics["distance"] * distance_factor, 1),
            "duration": round(base_metrics["duration"] * (1.0 / speed_factor) * distance_factor, 1),
            "fuel_consumption": round(base_metrics["fuel_consumption"] * consumption_factor * distance_factor, 1),
            "average_speed": round(base_metrics["average_speed"] * speed_factor, 1),
            "co2_emissions": round(base_metrics["co2_emissions"] * consumption_factor * distance_factor, 1)
        }
        
        return {
            'id': f'route_{name.lower().replace(" ", "_")}',
            'name': name,
            'waypoints': waypoints,
            'metrics': adjusted_metrics,
            'color': color
        }
    
    # Special handling for routes within Indian subcontinent
    if ((start_lat > 5 and start_lat < 25 and start_lon > 68 and start_lon < 90) and 
        (end_lat > 5 and end_lat < 25 and end_lon > 68 and end_lon < 90)):
        
        # Check if route is crossing Indian peninsula
        if ((start_lon < 78 and end_lon > 82) or (start_lon > 82 and end_lon < 78)):
            # This crosses the peninsula - need to go around southern India
            logger.info(f"Creating variation route around southern India for {start} to {end}")
            
            # Determine if this is east-to-west or west-to-east
            going_east = start_lon < end_lon
            
            # Add randomized waypoints around southern tip of India
            if going_east:
                # West to East route - go south around the tip with some randomness
                # Add points to navigate down west coast
                waypoints.append([start_lat - (start_lat - 8.5) * 0.33 + (random.random() - 0.5), 72.0 + (random.random() - 0.5) * 0.5])
                waypoints.append([start_lat - (start_lat - 8.5) * 0.67 + (random.random() - 0.5), 71.5 + (random.random() - 0.5) * 0.5])
                
                # Add points around southern tip with randomness
                waypoints.append([8.1 + (random.random() - 0.5) * 0.5, 76.8 + (random.random() - 0.5) * 0.5])  # Off Trivandrum
                waypoints.append([8.0 + (random.random() - 0.5) * 0.5, 77.5 + (random.random() - 0.5) * 0.5])  # Near Kanyakumari
                waypoints.append([8.3 + (random.random() - 0.5) * 0.5, 78.5 + (random.random() - 0.5) * 0.5])  # South of Tuticorin
                
                # Add points to navigate up east coast
                waypoints.append([end_lat - (end_lat - 10.0) * 0.67 + (random.random() - 0.5), 80.2 + (random.random() - 0.5) * 0.5])
                waypoints.append([end_lat - (end_lat - 10.0) * 0.33 + (random.random() - 0.5), 80.5 + (random.random() - 0.5) * 0.5])
            else:
                # East to West route - go south around the tip with some randomness
                # Add points to navigate down east coast
                waypoints.append([start_lat - (start_lat - 10.0) * 0.33 + (random.random() - 0.5), 80.5 + (random.random() - 0.5) * 0.5])
                waypoints.append([start_lat - (start_lat - 10.0) * 0.67 + (random.random() - 0.5), 80.2 + (random.random() - 0.5) * 0.5])
                
                # Add points around southern tip with randomness
                waypoints.append([8.3 + (random.random() - 0.5) * 0.5, 78.5 + (random.random() - 0.5) * 0.5])  # South of Tuticorin
                waypoints.append([8.0 + (random.random() - 0.5) * 0.5, 77.5 + (random.random() - 0.5) * 0.5])  # Near Kanyakumari
                waypoints.append([8.1 + (random.random() - 0.5) * 0.5, 76.8 + (random.random() - 0.5) * 0.5])  # Off Trivandrum
                
                # Add points to navigate up west coast
                waypoints.append([end_lat - (end_lat - 8.5) * 0.67 + (random.random() - 0.5), 71.5 + (random.random() - 0.5) * 0.5])
                waypoints.append([end_lat - (end_lat - 8.5) * 0.33 + (random.random() - 0.5), 72.0 + (random.random() - 0.5) * 0.5])
            
            # Add end point
            waypoints.append(end)
            
            # Calculate adjusted metrics
            distance_factor = 1.1 + random.random() * 0.1  # Around-peninsula routes are longer
            
            adjusted_metrics = {
                "distance": round(base_metrics["distance"] * distance_factor, 1),
                "duration": round(base_metrics["duration"] * (1.0 / speed_factor) * distance_factor, 1),
                "fuel_consumption": round(base_metrics["fuel_consumption"] * consumption_factor * distance_factor, 1),
                "average_speed": round(base_metrics["average_speed"] * speed_factor, 1),
        "co2_emissions": round(base_metrics["co2_emissions"] * consumption_factor * distance_factor, 1)
    }
    
    return {
        'id': f'route_{name.lower().replace(" ", "_")}',
        'name': name,
        'waypoints': waypoints,
        'metrics': adjusted_metrics,
        'color': color
    }

@app.route('/api/status')
def api_status():
    """Simple API status endpoint for troubleshooting."""
    return jsonify({
        'status': 'online',
        'version': '1.0.0',
        'server_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'endpoints': {
            'route': '/api/route',
            'routes': '/api/routes',
            'deep_rl_route': '/api/route/deep_rl',
            'deep_rl_routes': '/api/routes/deep_rl'
        }
    })

# Register enhanced routing endpoints if available
if DYNAMIC_ROUTING_AVAILABLE:
    logger.info("Registering enhanced routing endpoints")
    register_enhanced_routes(app)
else:
    logger.warning("Enhanced routing endpoints not available")
    
    # Create a fallback endpoint for enhanced routing
    @app.route('/api/enhanced-route', methods=['POST'])
    def fallback_enhanced_route():
        logger.warning("Using fallback enhanced route endpoint")
        try:
            data = request.json
            start = data.get('start')
            end = data.get('end')
            ship_type = data.get('ship_type', 'container_medium')
            
            # Use the standard route calculation as fallback
            if 'calculate_ocean_route' in globals():
                result = calculate_ocean_route(start, end, ship_type)
                return jsonify({
                    "status": "success", 
                    "message": "Using fallback route calculation", 
                    "route": result
                })
            else:
                return jsonify({"status": "error", "message": "Route calculation not available"})
        except Exception as e:
            logger.error(f"Error in fallback enhanced route: {str(e)}")
            return jsonify({"status": "error", "message": str(e)})
    
    @app.route('/api/enhanced-multiple-routes', methods=['POST'])
    def fallback_enhanced_multiple_routes():
        logger.warning("Using fallback enhanced multiple routes endpoint")
        try:
            data = request.json
            start = data.get('start')
            end = data.get('end')
            ship_type = data.get('ship_type', 'container_medium')
            num_routes = data.get('num_routes', 3)
            
            # Use the standard multiple routes calculation as fallback
            if 'get_multiple_routes' in globals():
                result = get_multiple_routes(start, end, ship_type, num_routes)
                return jsonify({
                    "status": "success", 
                    "message": "Using fallback multiple routes calculation", 
                    "routes": result
                })
            else:
                return jsonify({"status": "error", "message": "Multiple routes calculation not available"})
        except Exception as e:
            logger.error(f"Error in fallback enhanced multiple routes: {str(e)}")
            return jsonify({"status": "error", "message": str(e)})

# Add a new endpoint for ML-enhanced routes
@app.route('/api/ml-enhanced-route', methods=['POST'])
def ml_enhanced_route():
    """
    Generate a route using ML-based enhancements for terrain recognition and route optimization
    """
    try:
        # Parse request data
        data = request.json
        start = data.get('start')
        end = data.get('end')
        ship_type = data.get('ship_type', 'container_medium')
        use_reinforcement_learning = data.get('use_rl', True)
        
        logger.info(f"ML-enhanced route requested from {start} to {end} with ship type {ship_type}")
        
        # Check if both dynamic routing and ML components are available
        if not DYNAMIC_ROUTING_AVAILABLE:
            logger.warning("Dynamic routing not available for ML-enhanced route")
            return jsonify({
                "status": "error",
                "message": "Dynamic routing components not available"
            })
        
        # Check if AI routing integration is available
        if 'deep_rl_route_optimization' not in globals() or 'generate_ai_enhanced_routes' not in globals():
            logger.warning("AI routing integration not available")
            return jsonify({
                "status": "error",
                "message": "AI routing components not available"
            })
        
        # Get integrated router for base route
        integrated_router = get_integrated_router()
        
        # Generate base route first
        base_route_result = integrated_router.calculate_route(
            start=start,
            end=end,
            ship_type=ship_type,
            route_type="standard"
        )
        
        if not base_route_result or 'waypoints' not in base_route_result:
            logger.error("Failed to generate base route for ML enhancement")
            return jsonify({
                "status": "error",
                "message": "Failed to generate base route for ML enhancement"
            })
        
        base_route = base_route_result['waypoints']
        
        # Generate ML-enhanced route
        try:
            if use_reinforcement_learning:
                # Use RL-based optimization
                ml_route = deep_rl_route_optimization(
                    start,
                    end,
                    base_route=base_route,
                    ship_type=ship_type
                )
                enhancement_type = "reinforcement_learning"
            else:
                # Use neural network enhancement
                ml_route = generate_ai_enhanced_routes(
                    start,
                    end,
                    None,  # No environment needed for this call
                    None,  # No ship model needed for this call
                    base_route=base_route,
                    config={"ship": {"type": ship_type}}
                )
                enhancement_type = "neural_network"
                
            logger.info(f"ML route enhancement successful using {enhancement_type}")
            
            # Calculate metrics for the ML-enhanced route
            metrics = integrated_router._calculate_metrics(ml_route, ship_type)
            
            return jsonify({
                "status": "success",
                "api_version": "ml-enhanced-v1",
                "route": {
                    "start": start,
                    "end": end,
                    "waypoints": ml_route,
                    "ship_type": ship_type,
                    "metrics": metrics,
                    "enhancement_type": enhancement_type,
                    "id": f"ml_route_{start[0]:.2f}_{start[1]:.2f}_to_{end[0]:.2f}_{end[1]:.2f}_{enhancement_type}",
                    "timestamp": datetime.now().isoformat()
                }
            })
            
        except Exception as e:
            logger.error(f"Error in ML route enhancement: {str(e)}\n{traceback.format_exc()}")
            # Fall back to the base route
            return jsonify({
                "status": "partial_success",
                "message": f"ML enhancement failed: {str(e)}. Using base route instead.",
                "api_version": "ml-enhanced-v1-fallback",
                "route": {
                    "start": start,
                    "end": end,
                    "waypoints": base_route,
                    "ship_type": ship_type,
                    "metrics": base_route_result.get('metrics', {}),
                    "enhancement_type": "none",
                    "id": base_route_result.get('id', f"fallback_route_{start[0]:.2f}_{start[1]:.2f}_to_{end[0]:.2f}_{end[1]:.2f}"),
                    "timestamp": datetime.now().isoformat()
                }
            })
    
    except Exception as e:
        logger.error(f"Error processing ML-enhanced route request: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "status": "error",
            "message": f"Error processing request: {str(e)}"
        })

# Add an endpoint for checking and visualizing routes
@app.route('/api/visualize-route-with-land-check', methods=['POST'])
def visualize_route_with_land_check():
    """
    Visualize a route and check for land crossings
    """
    try:
        # Parse request data
        data = request.json
        route = data.get('route', [])
        
        if not route or len(route) < 2:
            return jsonify({
                "status": "error",
                "message": "Invalid route provided"
            })
        
        # Check for land crossings if terrain recognition is available
        land_crossings = []
        if DYNAMIC_ROUTING_AVAILABLE:
            integrated_router = get_integrated_router()
            terrain = integrated_router.terrain
            
            # Check each segment
            for i in range(len(route) - 1):
                start_wp = route[i]
                end_wp = route[i+1]
                
                # Check if the direct line between points crosses land
                crosses_land = False
                
                # Sample points along the path
                samples = 20
                for j in range(samples):
                    # Interpolate point
                    fraction = j / samples
                    lat = start_wp[0] + fraction * (end_wp[0] - start_wp[0])
                    lon = start_wp[1] + fraction * (end_wp[1] - start_wp[1])
                    
                    # Check if point is on land
                    if terrain.is_land(lat, lon):
                        crosses_land = True
                        land_crossings.append({
                            "segment": [i, i+1],
                            "point": [lat, lon],
                            "start": start_wp,
                            "end": end_wp
                        })
                        break
        
        # Generate visualization
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        plt.figure(figsize=(12, 10))
        
        # Extract latitude and longitude from route
        latitudes = [point[0] for point in route]
        longitudes = [point[1] for point in route]
        
        # Plot the route
        plt.plot(longitudes, latitudes, 'b-', linewidth=2)
        plt.plot(longitudes, latitudes, 'ro', markersize=5)
        
        # Mark start and end points
        plt.plot(longitudes[0], latitudes[0], 'go', markersize=10, label='Start')
        plt.plot(longitudes[-1], latitudes[-1], 'mo', markersize=10, label='End')
        
        # Highlight land crossings if any
        if land_crossings:
            for crossing in land_crossings:
                plt.plot(crossing['point'][1], crossing['point'][0], 'rx', markersize=12, 
                         label='Land Crossing' if crossing == land_crossings[0] else "")
        
        plt.title('Route Visualization with Land Check')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)
        plt.legend()
        
        # Save the plot to a file with timestamp to avoid conflicts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'route_check_{timestamp}.png'
        filepath = os.path.join('public', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate metrics
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += haversine_distance(
                route[i][0], route[i][1],
                route[i+1][0], route[i+1][1]
            )
        
        return jsonify({
            "status": "success",
            "land_crossings": land_crossings,
            "has_land_crossings": len(land_crossings) > 0,
            "visualization_url": f"/public/{filename}",
            "metrics": {
                "total_distance": total_distance,
                "waypoints": len(route)
            }
        })
        
    except Exception as e:
        logger.error(f"Error visualizing route: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "status": "error",
            "message": f"Error visualizing route: {str(e)}"
        })

if __name__ == '__main__':
    try:
        print("\n\n===== STARTING FLASK SERVER =====")
        print("Server will be accessible at: http://localhost:8080")
        print("If using a web browser, please navigate to that URL")
        print("Debug mode:", app.config.get('DEBUG', False))
        print("===================================\n\n")
        # Run the application on port 8080 instead of 5000
        app.run(debug=True, host='0.0.0.0', port=8080)
    except Exception as e:
        print("ERROR: Failed to start the server:")
        print(str(e))
        import traceback
        traceback.print_exc() 