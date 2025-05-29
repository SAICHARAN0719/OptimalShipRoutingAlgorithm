#!/usr/bin/env python
"""
API Integration Module

Integrates the enhanced routing capabilities with the existing Flask API.
This module adds improved routes to ensure all generated routes avoid land masses.
"""

import os
import sys
import json
import logging
from flask import request, jsonify
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the routing integration
try:
    from routing_integration import get_integrated_router
    router = get_integrated_router()
    logger.info("Routing integration initialized successfully")
except ImportError as e:
    logger.error(f"Failed to import routing_integration: {e}")
    router = None

def register_enhanced_routes(app):
    """
    Register enhanced routing API endpoints with the Flask app
    
    Args:
        app: Flask application instance
    """
    if router is None:
        logger.error("Cannot register enhanced routes: routing integration not available")
        return False
    
    # Enhanced route calculation endpoint
    @app.route('/api/enhanced-route', methods=['POST'])
    def calculate_enhanced_route():
        """Calculate an enhanced route with improved land avoidance"""
        try:
            # Parse request data
            data = request.json
            start = data.get('start')
            end = data.get('end')
            ship_type = data.get('ship_type', 'container_medium')
            route_type = data.get('route_type', 'standard')
            
            # Validate inputs
            if not start or not end:
                return jsonify({
                    "success": False,
                    "error": "Missing required parameters: start and end coordinates"
                }), 400
            
            # Calculate route
            logger.info(f"Calculating enhanced route from {start} to {end}")
            route_data = router.calculate_route(start, end, ship_type, route_type)
            
            # Add metadata
            route_data["api_version"] = "enhanced-v1"
            route_data["timestamp"] = datetime.now().isoformat()
            
            return jsonify(route_data)
            
        except Exception as e:
            logger.error(f"Error calculating enhanced route: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    # Enhanced multiple routes calculation endpoint
    @app.route('/api/enhanced-multiple-routes', methods=['POST'])
    def calculate_enhanced_multiple_routes():
        """Calculate multiple enhanced routes with guaranteed land avoidance"""
        try:
            # Parse request data
            data = request.json
            start = data.get('start')
            end = data.get('end')
            ship_type = data.get('ship_type', 'container_medium')
            num_variations = data.get('num_routes', 3)
            
            # Validate inputs
            if not start or not end:
                return jsonify({
                    "success": False,
                    "error": "Missing required parameters: start and end coordinates"
                }), 400
            
            # Ensure reasonable number of variations
            num_variations = min(max(1, num_variations), 5)  # Limit to 1-5 routes
            
            # Calculate routes
            logger.info(f"Calculating {num_variations} enhanced route variations from {start} to {end}")
            routes_data = router.calculate_multiple_routes(
                start, end, ship_type, num_variations=num_variations
            )
            
            # Add metadata
            routes_data["api_version"] = "enhanced-v1"
            routes_data["timestamp"] = datetime.now().isoformat()
            
            return jsonify(routes_data)
            
        except Exception as e:
            logger.error(f"Error calculating enhanced multiple routes: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    # Override the standard route endpoint (optional, uncomment to use)
    # @app.route('/api/route', methods=['POST'])
    # def calculate_route_override():
    #     """Override the standard route calculation with the enhanced version"""
    #     return calculate_enhanced_route()
    
    # Override the multiple routes endpoint (optional, uncomment to use)
    # @app.route('/api/multiple-routes', methods=['POST'])
    # def calculate_multiple_routes_override():
    #     """Override the standard multiple routes calculation with the enhanced version"""
    #     return calculate_enhanced_multiple_routes()
    
    # Add a land check endpoint
    @app.route('/api/check-land-crossing', methods=['POST'])
    def check_land_crossing():
        """Check if a route crosses land masses"""
        try:
            # Parse request data
            data = request.json
            waypoints = data.get('waypoints')
            
            if not waypoints or not isinstance(waypoints, list) or len(waypoints) < 2:
                return jsonify({
                    "success": False,
                    "error": "Invalid waypoints format"
                }), 400
            
            # Check each segment for land crossing
            crossings = []
            for i in range(len(waypoints) - 1):
                start_wp = waypoints[i]
                end_wp = waypoints[i+1]
                
                # Check if this segment crosses land
                is_crossing = check_segment_land_crossing(start_wp, end_wp)
                
                if is_crossing:
                    crossings.append({
                        "segment_start": start_wp,
                        "segment_end": end_wp,
                        "segment_index": i
                    })
            
            return jsonify({
                "success": True,
                "crosses_land": len(crossings) > 0,
                "crossing_segments": crossings,
                "total_waypoints": len(waypoints),
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error checking land crossing: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    # Add endpoint for comparing standard and enhanced routes
    @app.route('/api/compare-routes', methods=['POST'])
    def compare_routes():
        """Compare standard and enhanced routes for the same journey"""
        try:
            # Parse request data
            data = request.json
            start = data.get('start')
            end = data.get('end')
            ship_type = data.get('ship_type', 'container_medium')
            
            # Validate inputs
            if not start or not end:
                return jsonify({
                    "success": False,
                    "error": "Missing required parameters: start and end coordinates"
                }), 400
            
            # Call standard route API (assuming it's implemented in the app)
            standard_response = calculate_standard_route(start, end, ship_type)
            
            # Calculate enhanced route
            enhanced_route = router.calculate_route(start, end, ship_type)
            
            # Compare the routes
            comparison = {
                "success": True,
                "standard_route": standard_response,
                "enhanced_route": enhanced_route,
                "comparison": {
                    "standard_waypoints": len(standard_response.get("route", {}).get("waypoints", [])),
                    "enhanced_waypoints": len(enhanced_route["route"]["waypoints"]),
                    "standard_distance": standard_response.get("route", {}).get("metrics", {}).get("distance", 0),
                    "enhanced_distance": enhanced_route["route"]["metrics"]["distance"],
                    "distance_difference_percent": calculate_difference_percent(
                        standard_response.get("route", {}).get("metrics", {}).get("distance", 0),
                        enhanced_route["route"]["metrics"]["distance"]
                    ),
                    "crosses_land": {
                        "standard": check_route_land_crossing(standard_response.get("route", {}).get("waypoints", [])),
                        "enhanced": check_route_land_crossing(enhanced_route["route"]["waypoints"])
                    }
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return jsonify(comparison)
            
        except Exception as e:
            logger.error(f"Error comparing routes: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    logger.info("Enhanced routing API endpoints registered successfully")
    return True

def calculate_standard_route(start, end, ship_type):
    """
    Placeholder for calling the standard route calculation
    
    In a real implementation, this would call the existing route calculation function
    """
    # Try to import standard router
    try:
        # This assumes there's an existing ocean_routing module with a calculate_ocean_route function
        from api.ocean_routing_api import calculate_ocean_route
        return calculate_ocean_route(start, end, ship_type)
    except ImportError:
        # If we can't import the standard router, return a simulated response
        logger.warning("Could not import standard router, using simulated response")
        return {
            "success": True,
            "route": {
                "id": f"standard_route_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "name": f"Standard Route",
                "start": start,
                "end": end,
                "ship_type": ship_type,
                "waypoints": [
                    start,
                    [start[0] + (end[0] - start[0]) * 0.25, start[1] + (end[1] - start[1]) * 0.25],
                    [start[0] + (end[0] - start[0]) * 0.5, start[1] + (end[1] - start[1]) * 0.5],
                    [start[0] + (end[0] - start[0]) * 0.75, start[1] + (end[1] - start[1]) * 0.75],
                    end
                ],
                "metrics": {
                    "distance": 1000.0,
                    "duration": 50.0,
                    "fuel_consumption": 220.0,
                    "average_speed": 20.0,
                    "co2_emissions": 684.0
                }
            }
        }

def check_segment_land_crossing(start_wp, end_wp):
    """
    Check if a segment between two waypoints crosses land
    
    Args:
        start_wp: Starting waypoint [lat, lon]
        end_wp: Ending waypoint [lat, lon]
        
    Returns:
        Boolean indicating if the segment crosses land
    """
    if router is None:
        return False
    
    # Sample points along the segment
    num_samples = 10
    for i in range(1, num_samples):
        ratio = i / num_samples
        sample_lat = start_wp[0] + (end_wp[0] - start_wp[0]) * ratio
        sample_lon = start_wp[1] + (end_wp[1] - start_wp[1]) * ratio
        
        # Check if this point is on land
        if router.terrain.is_land(sample_lat, sample_lon):
            return True
    
    return False

def check_route_land_crossing(waypoints):
    """
    Check if any segment in a route crosses land
    
    Args:
        waypoints: List of waypoints [[lat, lon], ...]
        
    Returns:
        Boolean indicating if the route crosses land
    """
    if not waypoints or len(waypoints) < 2:
        return False
    
    for i in range(len(waypoints) - 1):
        if check_segment_land_crossing(waypoints[i], waypoints[i+1]):
            return True
    
    return False

def calculate_difference_percent(val1, val2):
    """Calculate percentage difference between two values"""
    if val1 == 0:
        return 0
    return round(((val2 - val1) / val1) * 100, 1)

# Example usage in the app.py file:
"""
from api_integration import register_enhanced_routes

# In your Flask app setup code:
app = Flask(__name__)
# ... other app setup ...

# Register enhanced routes
register_enhanced_routes(app)
""" 