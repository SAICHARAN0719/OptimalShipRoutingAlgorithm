import os
import sys
import logging
import argparse
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from geopy.distance import geodesic
import numpy as np

from models.ship_model import Ship
from models.environment_model import Environment
from algorithms.routing_algorithm import optimize_route
from simulation.simulate_voyage import simulate_voyage, calculate_fuel_reserves
from simulation.validation import validate_route, validate_weather_forecast
from visualization.plot_route import plot_route
from visualization.plot_metrics import plot_route_metrics
from utils.data_preprocessing import preprocess_data
from utils.logger import setup_logging

# Import dynamic routing components
try:
    from dynamic_routing import TerrainRecognition, AISDataIntegration, PathfindingAlgorithms, ShipRouting, CONFIG
    from routing_integration import RoutingIntegration, get_integrated_router
    from route_variations import RouteVariationGenerator
    # Flag to indicate dynamic routing is available
    DYNAMIC_ROUTING_AVAILABLE = True
    logging.info("Dynamic routing components loaded successfully")
except ImportError as e:
    logging.warning(f"Dynamic routing components not available: {str(e)}")
    DYNAMIC_ROUTING_AVAILABLE = False

# Import ML models if available
try:
    import tensorflow as tf
    from algorithms.ai_routing_integration import generate_ai_enhanced_routes
    ML_MODELS_AVAILABLE = True
    logging.info("Machine learning models loaded successfully")
except ImportError as e:
    logging.warning(f"Machine learning models not available: {str(e)}")
    ML_MODELS_AVAILABLE = False

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading configuration: {str(e)}")
        raise

def setup_logging(config: Dict[str, Any]):
    """Setup logging configuration."""
    try:
        log_config = config.get('logging', {})
        logging.basicConfig(
            level=log_config.get('level', 'INFO'),
            format=log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s'),
            filename=log_config.get('file', 'optimal_ship_routing.log')
        )
        logging.info("Logging configured successfully")
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        raise

def initialize_components(config: Dict[str, Any]) -> Tuple[Ship, Environment]:
    """Initialize ship and environment components."""
    try:
        # Initialize ship
        ship_config = config.get('ship', {})
        ship = Ship(ship_config)
        
        # Initialize environment
        env_config = config.get('environment', {})
        data_path = env_config.get('data_path', 'data/processed/environmental_data.nc')
        
        # Preprocess data if needed
        if not Path(data_path).exists():
            raw_data_path = env_config.get('raw_data_path', 'data/raw/incois_data.csv')
            preprocess_data(raw_data_path, data_path)
        
        environment = Environment(data_path)
        
        return ship, environment
        
    except Exception as e:
        logging.error(f"Error initializing components: {str(e)}")
        raise

def validate_inputs(start: Tuple[float, float], end: Tuple[float, float],
                   environment: Environment) -> bool:
    """Validate input parameters."""
    try:
        # Get environment bounds
        bounds = environment.get_bounds()
        lat_range = bounds['latitude']
        lon_range = bounds['longitude']
        
        # Check if points are within environment bounds
        for point, label in [(start, 'Start'), (end, 'End')]:
            lat, lon = point
            if not (lat_range[0] <= lat <= lat_range[1] and
                   lon_range[0] <= lon <= lon_range[1]):
                logging.error(f"{label} point {point} is outside environment bounds {lat_range}, {lon_range}")
                return False
        
        return True
        
    except Exception as e:
        logging.error(f"Error validating inputs: {str(e)}")
        return False

def optimize_ship_route(config_path: str,
                       start: Tuple[float, float],
                       end: Tuple[float, float],
                       start_time: datetime = None,
                       use_dynamic_routing: bool = True,
                       use_ml_enhancement: bool = False,
                       num_variations: int = 1) -> Dict[str, Any]:
    """
    Main function to optimize ship route.
    
    Args:
        config_path: Path to configuration file
        start: Start position (lat, lon)
        end: End position (lat, lon)
        start_time: Optional start time (defaults to current time)
        use_dynamic_routing: Whether to use enhanced dynamic routing with land avoidance
        use_ml_enhancement: Whether to use ML-enhanced routing
        num_variations: Number of route variations to generate
        
    Returns:
        Dictionary containing optimization results
    """
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Setup logging
        setup_logging(config)
        
        # Add debug logging
        logging.info(f"Starting route optimization from {start} to {end} at time {start_time}")
        
        # Initialize components
        ship, environment = initialize_components(config)
        
        # Log environment bounds
        logging.info(f"Environment bounds: Lat {environment.get_bounds()['latitude']}, Lon {environment.get_bounds()['longitude']}")
        
        # Validate inputs
        if not validate_inputs(start, end, environment):
            raise ValueError("Invalid input parameters")
        
        # Set default start time if not provided
        if start_time is None:
            start_time = datetime.now()
            
        logging.info(f"Using start time: {start_time}")
        
        # Validate weather forecast
        forecast_region = (
            min(start[0], end[0]) - 1.0,
            max(start[0], end[0]) + 1.0,
            min(start[1], end[1]) - 1.0,
            max(start[1], end[1]) + 1.0
        )
        
        # Use routing configuration for max duration
        max_days = config.get('routing', {}).get('max_duration', 10) / 24  # Convert hours to days, default 10 days
        
        forecast_time_range = (
            start_time,
            start_time + timedelta(days=max_days)
        )
        
        validation_result = validate_weather_forecast(
            environment, forecast_region, forecast_time_range
        )
        
        if not validation_result.get('passed', False):
            logging.warning("Weather forecast validation failed")
            
        # Use dynamic routing with land avoidance if enabled and available
        route_results = {}
        if use_dynamic_routing and DYNAMIC_ROUTING_AVAILABLE:
            logging.info("Using dynamic routing with land avoidance")
            
            # Get the integrated router
            integrated_router = get_integrated_router()
            
            if num_variations > 1:
                # Generate multiple route variations
                route_result = integrated_router.calculate_multiple_routes(
                    start=start,
                    end=end,
                    ship_type=config.get('ship', {}).get('type', 'container_medium'),
                    num_variations=num_variations
                )
                routes = [r['waypoints'] for r in route_result['routes']]
                primary_route = routes[0]  # Use the first variation as primary route
                
                # Store all variations
                route_results['variations'] = routes
                route_results['variation_metrics'] = [r['metrics'] for r in route_result['routes']]
                
                logging.info(f"Generated {len(routes)} route variations")
            else:
                # Generate single optimized route
                route_result = integrated_router.calculate_route(
                    start=start,
                    end=end,
                    ship_type=config.get('ship', {}).get('type', 'container_medium'),
                    route_type="standard"
                )
                primary_route = route_result['waypoints']
            
            # Use ML enhancement if requested and available
            if use_ml_enhancement and ML_MODELS_AVAILABLE:
                logging.info("Applying ML-based route enhancement")
                try:
                    ml_enhanced_route = generate_ai_enhanced_routes(
                        start, end, environment, ship, 
                        base_route=primary_route,
                        config=config
                    )
                    logging.info(f"ML enhancement applied successfully, new route has {len(ml_enhanced_route)} waypoints")
                    route = ml_enhanced_route
                    route_results['ml_enhanced'] = True
                except Exception as e:
                    logging.error(f"Error applying ML enhancement: {str(e)}")
                    route = primary_route
            else:
                route = primary_route
                
        else:
            # Use traditional routing algorithm
            logging.info("Using traditional routing algorithm")
            route = optimize_route(start, end, environment, ship, config)
            
        if not route:
            logging.warning("No path found")
            return {
                "status": "error",
                "message": "No path found",
                "route": None,
                "fuel_used": None,
                "time_taken": None
            }
            
        logging.info(f"Route optimization complete, found path with {len(route)} waypoints")
        
        # Print the route waypoints
        print("\n=== OPTIMAL ROUTE WAYPOINTS ===")
        print("Latitude, Longitude")
        for waypoint in route:
            print(f"{waypoint[0]:.4f}, {waypoint[1]:.4f}")
        print("===========================\n")
        
        # Simulate voyage with optimized route
        try:
            fuel_used, time_taken = simulate_voyage(route, environment, ship)
            logging.info(f"Voyage simulation: {fuel_used:.2f} tons of fuel, {time_taken:.2f} hours")
        except Exception as e:
            logging.error(f"Error simulating voyage: {str(e)}")
            return {
                "status": "error",
                "message": f"Error simulating voyage: {str(e)}",
                "route": route,
                "fuel_used": None,
                "time_taken": None
            }
            
        # Calculate fuel reserves
        try:
            fuel_reserves = calculate_fuel_reserves(route, environment, ship)
            logging.info(f"Fuel reserves needed: {fuel_reserves:.2f} tons")
        except Exception as e:
            logging.error(f"Error calculating fuel reserves: {str(e)}")
            fuel_reserves = None
            
        # Generate route visualization
        try:
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
            plt.plot(start[1], start[0], 'go', markersize=10, label='Start')
            plt.plot(end[1], end[0], 'mo', markersize=10, label='End')
            
            # If we have multiple routes, plot them with different colors
            if 'variations' in route_results and len(route_results['variations']) > 1:
                colors = ['g', 'c', 'm', 'y', 'k']
                for i, var_route in enumerate(route_results['variations'][1:]):  # Skip the first one (primary route)
                    if i < len(colors):  # Ensure we don't exceed available colors
                        var_lats = [point[0] for point in var_route]
                        var_lons = [point[1] for point in var_route]
                        plt.plot(var_lons, var_lats, f'{colors[i]}--', linewidth=1, alpha=0.7, 
                                label=f'Variation {i+2}')
            
            plt.title('Optimal Ship Route with Land Avoidance')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(True)
            plt.legend()
            
            # Add information to the plot
            route_type = "ML-Enhanced" if use_ml_enhancement and ML_MODELS_AVAILABLE else "Dynamic" if use_dynamic_routing else "Traditional"
            plt.figtext(0.5, 0.01, f"Route Type: {route_type} | Distance: {calculate_total_distance(route):.2f} nm | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                       ha='center', fontsize=10)
            
            # Save the plot to a file
            plot_path = 'route_visualization.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\nRoute visualization saved to {plot_path}\n")
        except Exception as e:
            logging.error(f"Error generating route visualization: {str(e)}")
        
        # Return results
        results = {
            "status": "success",
            "route": route,
            "fuel_used": fuel_used,
            "time_taken": time_taken,
            "fuel_reserves": fuel_reserves,
            "validation": validation_result
        }
        
        # Add route variations if available
        if 'variations' in route_results:
            results['route_variations'] = route_results['variations']
            results['variation_metrics'] = route_results['variation_metrics']
        
        print("\n=== ROUTE SUMMARY ===")
        print(f"Number of waypoints: {len(route)}")
        print(f"Total distance: {calculate_total_distance(route):.2f} nautical miles")
        print(f"Estimated fuel consumption: {fuel_used:.2f} tons")
        print(f"Estimated travel time: {time_taken:.2f} hours ({time_taken/24:.2f} days)")
        print("=====================\n")
        
        return results
            
    except Exception as e:
        logging.error(f"Error in route optimization: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "route": None,
            "fuel_used": None,
            "time_taken": None
        }

def calculate_total_distance(route):
    """Calculate the total distance of a route in nautical miles."""
    total_distance = 0
    for i in range(len(route) - 1):
        point1 = route[i]
        point2 = route[i + 1]
        # Calculate distance in nautical miles (nm)
        distance = geodesic(point1, point2).nautical
        total_distance += distance
    return total_distance

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimal Ship Routing System')
    parser.add_argument('config', type=str, help='Path to configuration file')
    parser.add_argument('--start', type=float, nargs=2, required=True, 
                        help='Start position as latitude longitude')
    parser.add_argument('--end', type=float, nargs=2, required=True,
                        help='End position as latitude longitude')
    parser.add_argument('--time', type=str, required=False, default=None,
                        help='Start time in format YYYY-MM-DD HH:MM:SS')
    
    args = parser.parse_args()
    
    start_pos = (args.start[0], args.start[1])
    end_pos = (args.end[0], args.end[1])
    
    if args.time:
        start_time = datetime.strptime(args.time, '%Y-%m-%d %H:%M:%S')
    else:
        start_time = None
    
    try:
        results = optimize_ship_route(
            args.config,
            start_pos,
            end_pos,
            start_time
        )
        
        # Print results summary
        if results['status'] == 'success':
            print(f"\nOptimization completed successfully.")
        else:
            print(f"\nOptimization failed: {results['message']}")
            
    except Exception as e:
        import traceback
        print(f"Error executing route optimization:")
        print(traceback.format_exc())
        sys.exit(1)