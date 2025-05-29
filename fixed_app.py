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
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import yaml

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
    # Define fallback classes in case imports fail
    class Ship:
        def __init__(self, length=200.0, weight=50000.0, fuel_rate=0.3, beam=None, draft=None, max_speed=25.0, current_fuel=None, ship_type=None, config=None):
            self.length = length
            self.weight = weight
            self.fuel_rate = fuel_rate
            self.beam = beam
            self.draft = draft
            self.max_speed = max_speed
            self.current_fuel = current_fuel
            self.max_wave_height = 5.0
            self.max_wind_speed = 40.0
            self.service_speed = max_speed * 0.8
            self.specific_fuel_consumption = 180
            self.max_power = 15000
    
    class Environment:
        def __init__(self, data_path=None, config=None):
            self.data_path = data_path
            self.config = config or {}
        
        def get_bounds(self):
            return {
                'latitude': (0, 90),
                'longitude': (-180, 180),
                'time': (datetime.now() - timedelta(days=30), datetime.now() + timedelta(days=30))
            }
        
        def get_conditions(self, position, time):
            lat, lon = position
            return {
                'wave_height': 1.0,
                'wind_speed': 10.0,
                'current_speed': 0.5,
                'current_direction': 0.0
            }
        
        def is_safe_conditions(self, conditions):
            return conditions['wave_height'] < 4.0 and conditions['wind_speed'] < 30.0

# Import ocean routing API - Fix import names to match what's available
try:
    from api.ocean_routing_api import calculate_ocean_route, get_multiple_routes, calculate_ship_parameters, calculate_route_metrics
    logger.info("Successfully imported ocean routing API")
except Exception as e:
    logger.error(f"Error importing ocean routing API: {e}\n{traceback.format_exc()}")
    # Define fallback functions if imports fail
    def calculate_ship_parameters(ship_type):
        # Default parameters for different ship types
        ship_params = {
            'container_small': {
                'max_speed_knots': 18,
                'cruising_speed_knots': 14,
                'fuel_capacity_tons': 1000,
                'fuel_consumption_rate': 0.08
            },
            'container_medium': {
                'max_speed_knots': 22,
                'cruising_speed_knots': 16,
                'fuel_capacity_tons': 2500,
                'fuel_consumption_rate': 0.12
            },
            'container_large': {
                'max_speed_knots': 25,
                'cruising_speed_knots': 18,
                'fuel_capacity_tons': 4000,
                'fuel_consumption_rate': 0.15
            },
            'tanker': {
                'max_speed_knots': 15,
                'cruising_speed_knots': 12,
                'fuel_capacity_tons': 3000,
                'fuel_consumption_rate': 0.14
            },
            'bulk_carrier': {
                'max_speed_knots': 14,
                'cruising_speed_knots': 11,
                'fuel_capacity_tons': 2500,
                'fuel_consumption_rate': 0.11
            },
            'default': {
                'max_speed_knots': 20,
                'cruising_speed_knots': 15,
                'fuel_capacity_tons': 2000,
                'fuel_consumption_rate': 0.1
            }
        }
        return ship_params.get(ship_type, ship_params['default'])
    
    def calculate_route_metrics(waypoints, ship_params):
        if not waypoints or len(waypoints) < 2:
            return {
                "distance": 0,
                "duration": 0,
                "fuel_consumption": 0,
                "average_speed": 0,
                "co2_emissions": 0
            }
        
        # Extract parameters
        cruising_speed = ship_params.get('cruising_speed_knots', 15)
        fuel_rate = ship_params.get('fuel_consumption_rate', 0.1)
        
        # Calculate total distance
        total_distance = 0
        for i in range(1, len(waypoints)):
            start_lat, start_lon = waypoints[i-1]
            end_lat, end_lon = waypoints[i]
            
            # Calculate distance using Haversine formula
            segment_distance = haversine_distance(start_lat, start_lon, end_lat, end_lon)
            total_distance += segment_distance
        
        # Calculate duration in hours
        duration = total_distance / cruising_speed
        
        # Calculate fuel consumption
        fuel_consumption = total_distance * fuel_rate
        
        # Calculate CO2 emissions (tons) - Assuming 3.2 tons of CO2 per ton of fuel
        co2_emissions = fuel_consumption * 3.2
        
        return {
            "distance": round(total_distance, 1),
            "duration": round(duration, 1),
            "fuel_consumption": round(fuel_consumption, 1),
            "average_speed": cruising_speed,
            "co2_emissions": round(co2_emissions, 1)
        }
    
    def get_multiple_routes(*args, **kwargs):
        # Implement a simple fallback for multiple routes
        start_point = kwargs.get('start_point')
        end_point = kwargs.get('end_point')
        ship_type = kwargs.get('ship_type', 'container_medium')
        
        if not start_point or not end_point:
            return {"error": "Start and end points are required"}
        
        # Create direct route
        direct_route = create_direct_route(start_point, end_point, ship_type)
        
        # Modify the route slightly to create alternatives
        routes = [
            direct_route,
            create_variation_route(start_point, end_point, ship_type, "Eco-Friendly", 0.8, 0.7, "#4CAF50"),
            create_variation_route(start_point, end_point, ship_type, "Fast", 1.2, 1.3, "#F44336")
        ]
        
        return {
            "success": True,
            "count": len(routes),
            "routes": routes
        }
    
    def create_direct_route(start, end, ship_type):
        # Get ship parameters
        ship_params = calculate_ship_parameters(ship_type)
        
        # Create waypoints
        waypoints = [start, end]
        
        # Calculate metrics
        metrics = calculate_route_metrics(waypoints, ship_params)
        
        return {
            "id": "direct_" + datetime.now().strftime('%Y%m%d_%H%M%S'),
            "name": "Direct Route",
            "waypoints": waypoints,
            "metrics": metrics,
            "route_type": "Optimal",
            "color": "#3388FF",
            "ship_type": ship_type
        }
    
    def create_variation_route(start, end, ship_type, name, speed_factor, fuel_factor, color):
        # Get ship parameters
        base_params = calculate_ship_parameters(ship_type)
        
        # Create modified parameters
        modified_params = base_params.copy()
        modified_params['cruising_speed_knots'] = base_params['cruising_speed_knots'] * speed_factor
        modified_params['fuel_consumption_rate'] = base_params['fuel_consumption_rate'] * fuel_factor
        
        # Create waypoints - for a variation, add a midpoint with slight offset
        lat1, lon1 = start
        lat2, lon2 = end
        
        # Create a midpoint with a slight deviation
        mid_lat = (lat1 + lat2) / 2 + random.uniform(-0.5, 0.5)
        mid_lon = (lon1 + lon2) / 2 + random.uniform(-0.5, 0.5)
        
        waypoints = [start, [mid_lat, mid_lon], end]
        
        # Calculate metrics
        metrics = calculate_route_metrics(waypoints, modified_params)
        
        return {
            "id": f"{name.lower()}_" + datetime.now().strftime('%Y%m%d_%H%M%S'),
            "name": name,
            "waypoints": waypoints,
            "metrics": metrics,
            "route_type": name,
            "color": color,
            "ship_type": ship_type
        }

app = Flask(__name__, static_folder='public')
CORS(app)

# Load configuration
try:
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
except Exception as e:
    logger.error(f"Error loading config: {e}")
    # Create default config
    config = {
        'environment': {
            'data_path': 'data/environment_data.csv'
        },
        'ship': {
            'length': 200.0,
            'weight': 50000.0,
            'fuel_rate': 0.3,
            'max_speed': 25.0
        }
    }

# Initialize environment and ship models
try:
    # Initialize the environment
    environment = Environment(config=config['environment'])
    logger.info("Environment initialized")
    
    # Create Ship instance with proper parameters
    ship = Ship(config=config['ship'])
    logger.info(f"Ship initialized with max speed {ship.max_speed} knots")
except Exception as e:
    logger.error(f"Error in initialization process: {e}\n{traceback.format_exc()}")
    # Create default environment and ship
    environment = Environment()
    ship = Ship()
    logger.info("Using default environment and ship due to initialization error")

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
        # Try to use the ocean routing API first
        try:
            # Get ship parameters
            ship_params = calculate_ship_parameters(ship_type)
            
            # Call the API with correct parameters
            result = calculate_ocean_route(
                start_point=start,
                end_point=end,
                ship_type=ship_type,
                route_type='efficient',
                include_metrics=True,
                use_weather=True
            )
            
            # Check if we got a valid result with waypoints
            if result and 'waypoints' in result and result['waypoints']:
                logger.info(f"Generated route with {len(result['waypoints'])} waypoints")
                return jsonify(result)
            else:
                logger.warning("Ocean routing API returned invalid result, falling back to direct route")
                # Fall through to direct route calculation
        except Exception as api_error:
            logger.error(f"Error using ocean routing API: {api_error}")
            # Fall through to direct route calculation
        
        # Fallback to direct route calculation
        logger.info("Calculating direct route as fallback")
        
        # Create a direct route between start and end
        direct_route = [start, end]
        
        # Calculate the distance
        lat1, lon1 = start
        lat2, lon2 = end
        distance = haversine_distance(lat1, lon1, lat2, lon2)
        
        # Get ship parameters
        ship_params = calculate_ship_parameters(ship_type)
        
        # Calculate metrics
        cruising_speed = ship_params.get('cruising_speed_knots', 15)
        duration = distance / cruising_speed
        fuel_rate = ship_params.get('fuel_consumption_rate', 0.1)
        fuel_consumption = distance * fuel_rate
        co2_emissions = fuel_consumption * 3.2
        
        # Create result with direct route
        result = {
            'success': True,
            'waypoints': direct_route,
            'metrics': {
                'distance': round(distance, 1),
                'duration': round(duration, 1),
                'fuel_consumption': round(fuel_consumption, 1),
                'average_speed': round(cruising_speed, 1),
                'co2_emissions': round(co2_emissions, 1)
            }
        }
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in route calculation: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e),
            'waypoints': [start, end],
            'metrics': {
                'distance': 0,
                'duration': 0,
                'fuel_consumption': 0,
                'average_speed': 0,
                'co2_emissions': 0
            }
        })

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
    
    # Validate required parameters
    if not start:
        return jsonify({'success': False, 'error': 'Start point is required'}), 400
    
    if not end:
        return jsonify({'success': False, 'error': 'End point is required'}), 400
    
    try:
        # Calculate multiple routes
        result = get_multiple_routes(
            start_point=start,
            end_point=end,
            ship_type=ship_type
        )
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error calculating multiple routes: {str(e)}\n{traceback.format_exc()}")
        
        # Create fallback routes
        direct_route = create_direct_route(start, end, ship_type)
        eco_route = create_variation_route(start, end, ship_type, "Eco-Friendly", 0.8, 0.7, "#4CAF50")
        fast_route = create_variation_route(start, end, ship_type, "Fast", 1.2, 1.3, "#F44336")
        
        fallback_result = {
            'success': False,
            'error': str(e),
            'count': 3,
            'routes': [direct_route, eco_route, fast_route]
        }
        
        return jsonify(fallback_result)

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
        # Try to use the Deep RL router if available
        try:
            # Initialize environment and ship models
            env_model = Environment(config=config['environment'])
            ship_model = Ship(ship_type=ship_type, config=config['ship'])
            
            # Calculate route using deep reinforcement learning
            logger.info(f"Calculating deep RL route from {start} to {end} with {ship_type}")
            route = deep_rl_route_optimization(
                start=start,
                end=end,
                env=env_model,
                ship=ship_model,
                config=config,
                departure_time=datetime.now()
            )
            
            if route and len(route) > 0:
                # Calculate route metrics
                ship_params = calculate_ship_parameters(ship_type)
                metrics = calculate_route_metrics(route, ship_params)
                
                # Format response
                result = {
                    'success': True,
                    'route': {
                        'id': 'deeprl_' + datetime.now().strftime('%Y%m%d_%H%M%S'),
                        'name': f"Deep RL Route {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        'waypoints': route,
                        'metrics': metrics,
                        'ship_type': ship_type
                    }
                }
                
                return jsonify(result)
            else:
                logger.warning("Deep RL routing returned no route, falling back to direct route")
                # Fall through to direct route calculation
        except Exception as rl_error:
            logger.error(f"Error using Deep RL router: {rl_error}")
            # Fall through to direct route calculation
        
        # Fallback to direct route calculation
        logger.info("Calculating direct route as fallback for Deep RL")
        
        # Create a direct route between start and end
        direct_route = [start, end]
        
        # Calculate the distance
        lat1, lon1 = start
        lat2, lon2 = end
        distance = haversine_distance(lat1, lon1, lat2, lon2)
        
        # Get ship parameters
        ship_params = calculate_ship_parameters(ship_type)
        
        # Calculate metrics
        cruising_speed = ship_params.get('cruising_speed_knots', 15)
        duration = distance / cruising_speed
        fuel_rate = ship_params.get('fuel_consumption_rate', 0.1)
        fuel_consumption = distance * fuel_rate
        co2_emissions = fuel_consumption * 3.2
        
        # Create result with direct route
        result = {
            'success': True,
            'route': {
                'id': 'direct_' + datetime.now().strftime('%Y%m%d_%H%M%S'),
                'name': f"Direct Route {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                'waypoints': direct_route,
                'metrics': {
                    'distance': round(distance, 1),
                    'duration': round(duration, 1),
                    'fuel_consumption': round(fuel_consumption, 1),
                    'average_speed': round(cruising_speed, 1),
                    'co2_emissions': round(co2_emissions, 1)
                },
                'ship_type': ship_type
            }
        }
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in Deep RL route calculation: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

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
        # Try to use the Deep RL router for multiple routes if available
        try:
            # Initialize environment and ship models
            env_model = Environment(config=config['environment'])
            ship_model = Ship(ship_type=ship_type, config=config['ship'])
            
            # Calculate multiple routes using deep reinforcement learning
            logger.info(f"Calculating {num_routes} deep RL routes from {start} to {end} with {ship_type}")
            routes = generate_deep_rl_routes(
                start=start,
                end=end,
                env=env_model,
                ship=ship_model,
                config=config,
                departure_time=datetime.now(),
                num_routes=num_routes
            )
            
            if routes and len(routes) > 0:
                # Format response
                result = {
                    'success': True,
                    'count': len(routes),
                    'routes': routes
                }
                
                return jsonify(result)
            else:
                logger.warning("Deep RL routing returned no routes, falling back to standard variations")
                # Fall through to fallback route generation
        except Exception as rl_error:
            logger.error(f"Error using Deep RL router for multiple routes: {rl_error}")
            # Fall through to fallback route generation
        
        # Fallback to generating route variations
        logger.info("Generating route variations as fallback for Deep RL multiple routes")
        
        # Create a direct route and variations
        direct_route = create_direct_route(start, end, ship_type)
        
        # Define variation parameters for different route types
        variations = [
            {"name": "Eco-Friendly", "speed_factor": 0.8, "fuel_factor": 0.7, "color": "#4CAF50"},
            {"name": "Fast", "speed_factor": 1.2, "fuel_factor": 1.3, "color": "#F44336"},
            {"name": "Safe", "speed_factor": 0.9, "fuel_factor": 1.1, "color": "#2196F3"}
        ]
        
        # Create variation routes
        variation_routes = [
            create_variation_route(
                start, end, ship_type, 
                var["name"], var["speed_factor"], var["fuel_factor"], var["color"]
            )
            for var in variations[:min(num_routes-1, len(variations))]
        ]
        
        # Combine direct route with variations to get desired number of routes
        all_routes = [direct_route] + variation_routes
        
        # Format response
        result = {
            'success': True,
            'count': len(all_routes),
            'routes': all_routes
        }
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error calculating multiple Deep RL routes: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 