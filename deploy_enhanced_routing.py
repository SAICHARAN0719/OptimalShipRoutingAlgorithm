#!/usr/bin/env python
"""
Deploy Enhanced Routing

This script demonstrates how to integrate the enhanced routing system
with the existing application. It creates a new Flask application that
uses the enhanced routing system for all routes.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='deploy_enhanced_routing.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

# Add console handler to see logs in real-time
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Deploy the enhanced routing system")
    parser.add_argument(
        "--port", type=int, default=5000,
        help="Port to run the server on (default: 5000)"
    )
    parser.add_argument(
        "--mode", type=str, choices=["standalone", "integrated"], default="integrated",
        help="Deployment mode: standalone (new server) or integrated (with existing app)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Run the server in debug mode"
    )
    parser.add_argument(
        "--override", action="store_true",
        help="Override existing API endpoints"
    )
    return parser.parse_args()

def create_standalone_app():
    """Create a standalone Flask application with the enhanced routing system"""
    from flask import Flask, jsonify, request
    from api_integration import register_enhanced_routes
    
    app = Flask(__name__)
    
    # Register the enhanced routing endpoints
    success = register_enhanced_routes(app)
    
    if not success:
        logger.error("Failed to register enhanced routing endpoints")
        sys.exit(1)
    
    # Add basic routes
    @app.route('/')
    def index():
        """Root endpoint"""
        return jsonify({
            "name": "Enhanced Maritime Routing API",
            "version": "1.0.0",
            "description": "API for maritime routing with advanced land avoidance",
            "endpoints": [
                {
                    "path": "/api/enhanced-route",
                    "method": "POST",
                    "description": "Calculate an enhanced route with improved land avoidance"
                },
                {
                    "path": "/api/enhanced-multiple-routes",
                    "method": "POST",
                    "description": "Calculate multiple enhanced routes with guaranteed land avoidance"
                },
                {
                    "path": "/api/check-land-crossing",
                    "method": "POST",
                    "description": "Check if a route crosses land masses"
                },
                {
                    "path": "/api/compare-routes",
                    "method": "POST",
                    "description": "Compare standard and enhanced routes for the same journey"
                }
            ],
            "timestamp": datetime.now().isoformat()
        })
    
    # Add example route
    @app.route('/example')
    def example():
        """Example route calculation"""
        from routing_integration import get_integrated_router
        
        # Visakhapatnam to Shanghai
        start = [17.6868, 83.2185]
        end = [31.2304, 121.4737]
        
        # Get router and calculate route
        router = get_integrated_router()
        route = router.calculate_route(start, end)
        
        return jsonify(route)
    
    return app

def integrate_with_existing_app():
    """Integrate the enhanced routing system with the existing app"""
    # Try to import the existing app
    try:
        # This assumes app.py defines a Flask app instance called 'app'
        from app import app as existing_app
        
        # Register the enhanced routing endpoints
        from api_integration import register_enhanced_routes
        
        # Register the enhanced routes
        success = register_enhanced_routes(existing_app)
        
        if not success:
            logger.error("Failed to register enhanced routing endpoints")
            sys.exit(1)
        
        logger.info("Successfully integrated enhanced routing with existing app")
        
        return existing_app
        
    except ImportError as e:
        logger.error(f"Failed to import existing app: {e}")
        logger.info("Falling back to standalone mode")
        return create_standalone_app()

def patch_existing_routes(app, override=False):
    """Patch existing route endpoints with enhanced versions"""
    if not override:
        logger.info("Not overriding existing routes (use --override to enable)")
        return
    
    # Import necessary modules
    try:
        from flask import request, jsonify
        from routing_integration import get_integrated_router
        
        # Get the router
        router = get_integrated_router()
        
        # Get all routes/endpoints defined in the app
        routes = [rule.endpoint for rule in app.url_map.iter_rules()]
        
        # Check if the standard route endpoints exist
        if 'calculate_route' in routes:
            # This assumes the existing endpoint is defined like this in app.py:
            # @app.route('/api/route', methods=['POST'])
            # def calculate_route():
            #     ...
            
            # Override the route calculation
            logger.info("Overriding standard route calculation endpoint")
            
            # Store the original function
            original_calculate_route = app.view_functions['calculate_route']
            
            # Define the override function
            def calculate_route_override():
                """Enhanced route calculation that avoids land masses"""
                try:
                    # Parse request data
                    data = request.json
                    start = data.get('start')
                    end = data.get('end')
                    ship_type = data.get('ship_type', 'container_medium')
                    
                    # Check if we should use the enhanced version
                    use_enhanced = data.get('use_enhanced', True)
                    
                    if not use_enhanced:
                        # Use the original calculation
                        return original_calculate_route()
                    
                    # Use the enhanced calculation
                    route_data = router.calculate_route(start, end, ship_type)
                    
                    # Add original API compatibility wrapper if needed
                    # ... implementation depends on the original API format
                    
                    return jsonify(route_data)
                    
                except Exception as e:
                    logger.error(f"Error in overridden route calculation: {e}")
                    # Fall back to original calculation
                    return original_calculate_route()
            
            # Replace the view function
            app.view_functions['calculate_route'] = calculate_route_override
        
        # Similarly for multiple routes if it exists
        if 'calculate_multiple_routes' in routes:
            logger.info("Overriding multiple routes calculation endpoint")
            
            # Store the original function
            original_multiple_routes = app.view_functions['calculate_multiple_routes']
            
            # Define the override function
            def multiple_routes_override():
                """Enhanced multiple routes calculation that avoids land masses"""
                try:
                    # Parse request data
                    data = request.json
                    start = data.get('start')
                    end = data.get('end')
                    ship_type = data.get('ship_type', 'container_medium')
                    num_routes = data.get('num_routes', 3)
                    
                    # Check if we should use the enhanced version
                    use_enhanced = data.get('use_enhanced', True)
                    
                    if not use_enhanced:
                        # Use the original calculation
                        return original_multiple_routes()
                    
                    # Use the enhanced calculation
                    routes_data = router.calculate_multiple_routes(
                        start, end, ship_type, num_variations=num_routes
                    )
                    
                    # Add original API compatibility wrapper if needed
                    # ... implementation depends on the original API format
                    
                    return jsonify(routes_data)
                    
                except Exception as e:
                    logger.error(f"Error in overridden multiple routes calculation: {e}")
                    # Fall back to original calculation
                    return original_multiple_routes()
            
            # Replace the view function
            app.view_functions['calculate_multiple_routes'] = multiple_routes_override
            
        logger.info("Route overrides applied successfully")
        
    except Exception as e:
        logger.error(f"Failed to override existing routes: {e}")
        logger.error("Continuing without overriding")

def add_test_page(app):
    """Add a test page to the application"""
    from flask import send_from_directory, render_template_string
    
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create a simple test HTML file
    test_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced Maritime Routing Test</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; }
            .container { display: flex; }
            .input-panel { flex: 1; padding: 10px; }
            .map-panel { flex: 2; padding: 10px; }
            .form-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, select { width: 100%; padding: 8px; box-sizing: border-box; }
            button { background-color: #3498db; color: white; border: none; padding: 10px 15px; cursor: pointer; }
            #map { height: 500px; border: 1px solid #ddd; }
            #results { margin-top: 20px; }
            pre { background-color: #f8f9fa; padding: 10px; overflow: auto; }
        </style>
    </head>
    <body>
        <h1>Enhanced Maritime Routing Test</h1>
        
        <div class="container">
            <div class="input-panel">
                <h2>Route Parameters</h2>
                
                <div class="form-group">
                    <label for="start-lat">Start Latitude:</label>
                    <input type="number" id="start-lat" step="0.0001" value="17.6868">
                </div>
                
                <div class="form-group">
                    <label for="start-lon">Start Longitude:</label>
                    <input type="number" id="start-lon" step="0.0001" value="83.2185">
                </div>
                
                <div class="form-group">
                    <label for="end-lat">End Latitude:</label>
                    <input type="number" id="end-lat" step="0.0001" value="31.2304">
                </div>
                
                <div class="form-group">
                    <label for="end-lon">End Longitude:</label>
                    <input type="number" id="end-lon" step="0.0001" value="121.4737">
                </div>
                
                <div class="form-group">
                    <label for="ship-type">Ship Type:</label>
                    <select id="ship-type">
                        <option value="container_small">Container Ship (Small)</option>
                        <option value="container_medium" selected>Container Ship (Medium)</option>
                        <option value="container_large">Container Ship (Large)</option>
                        <option value="tanker_medium">Tanker (Medium)</option>
                        <option value="tanker_large">Tanker (Large)</option>
                        <option value="bulk_carrier">Bulk Carrier</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="num-routes">Number of Routes:</label>
                    <input type="number" id="num-routes" min="1" max="5" value="3">
                </div>
                
                <div class="form-group">
                    <button id="calculate-btn">Calculate Routes</button>
                    <button id="clear-btn">Clear</button>
                </div>
                
                <div id="status"></div>
            </div>
            
            <div class="map-panel">
                <h2>Route Visualization</h2>
                <div id="map"></div>
                
                <div id="results">
                    <h3>Route Details</h3>
                    <pre id="route-details">Calculate a route to see details...</pre>
                </div>
            </div>
        </div>
        
        <!-- Load the required scripts -->
        <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
        
        <script>
            // Initialize map
            const map = L.map('map').setView([20, 100], 4);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            }).addTo(map);
            
            // Keep track of all route layers
            let routeLayers = [];
            
            // Add event listeners
            document.getElementById('calculate-btn').addEventListener('click', calculateRoutes);
            document.getElementById('clear-btn').addEventListener('click', clearMap);
            
            // Function to calculate routes
            async function calculateRoutes() {
                const startLat = parseFloat(document.getElementById('start-lat').value);
                const startLon = parseFloat(document.getElementById('start-lon').value);
                const endLat = parseFloat(document.getElementById('end-lat').value);
                const endLon = parseFloat(document.getElementById('end-lon').value);
                const shipType = document.getElementById('ship-type').value;
                const numRoutes = parseInt(document.getElementById('num-routes').value);
                
                // Validate inputs
                if (isNaN(startLat) || isNaN(startLon) || isNaN(endLat) || isNaN(endLon)) {
                    setStatus('Error: Invalid coordinates', 'error');
                    return;
                }
                
                // Clear previous routes
                clearMap();
                
                // Show loading status
                setStatus('Calculating routes...', 'loading');
                
                try {
                    // Call the API for multiple routes
                    const response = await fetch('/api/enhanced-multiple-routes', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            start: [startLat, startLon],
                            end: [endLat, endLon],
                            ship_type: shipType,
                            num_routes: numRoutes
                        })
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    
                    const data = await response.json();
                    
                    // Display route details
                    document.getElementById('route-details').textContent = JSON.stringify(data, null, 2);
                    
                    // Display routes on map
                    if (data.success && data.routes && data.routes.length > 0) {
                        // Define colors for different routes
                        const colors = ['red', 'blue', 'green', 'purple', 'orange'];
                        
                        // Add each route to the map
                        data.routes.forEach((route, index) => {
                            const color = colors[index % colors.length];
                            const waypoints = route.waypoints.map(wp => [wp[0], wp[1]]);
                            
                            // Create polyline for the route
                            const routeLine = L.polyline(waypoints, {
                                color: color,
                                weight: 3,
                                opacity: 0.7
                            }).addTo(map);
                            
                            // Add start and end markers
                            const startMarker = L.marker(waypoints[0]).addTo(map);
                            const endMarker = L.marker(waypoints[waypoints.length - 1]).addTo(map);
                            
                            // Add to route layers for later clearing
                            routeLayers.push(routeLine, startMarker, endMarker);
                            
                            // Add a popup with route information
                            routeLine.bindPopup(
                                `<b>${route.name}</b><br>` +
                                `Distance: ${route.metrics.distance} nm<br>` +
                                `Duration: ${route.metrics.duration} hours<br>` +
                                `Waypoints: ${route.waypoints.length}`
                            );
                        });
                        
                        // Fit map to show all routes
                        const allWaypoints = data.routes.flatMap(route => route.waypoints);
                        const allLatLngs = allWaypoints.map(wp => [wp[0], wp[1]]);
                        map.fitBounds(allLatLngs);
                        
                        setStatus(`Successfully calculated ${data.routes.length} routes`, 'success');
                    } else {
                        setStatus('No routes found in response', 'error');
                    }
                } catch (error) {
                    console.error('Error calculating routes:', error);
                    setStatus(`Error: ${error.message}`, 'error');
                }
            }
            
            // Function to clear the map
            function clearMap() {
                // Remove all route layers
                routeLayers.forEach(layer => map.removeLayer(layer));
                routeLayers = [];
                
                // Clear route details
                document.getElementById('route-details').textContent = 'Calculate a route to see details...';
                
                setStatus('Map cleared', 'info');
            }
            
            // Function to set status message
            function setStatus(message, type) {
                const statusElement = document.getElementById('status');
                statusElement.textContent = message;
                
                // Set color based on message type
                statusElement.style.color = type === 'error' ? 'red' :
                                          type === 'success' ? 'green' :
                                          type === 'loading' ? 'blue' : 'black';
            }
        </script>
    </body>
    </html>
    """
    
    # Write the HTML file
    with open('templates/test_routing.html', 'w') as f:
        f.write(test_html)
    
    # Add route for the test page
    @app.route('/test-routing')
    def test_routing():
        return render_template_string(test_html)
    
    # Add route for static files
    @app.route('/static/<path:filename>')
    def serve_static(filename):
        return send_from_directory('static', filename)
    
    logger.info("Added test page at /test-routing")

def main():
    """Main function to deploy the enhanced routing system"""
    args = parse_arguments()
    
    logger.info(f"Deploying enhanced routing system in {args.mode} mode")
    
    # Create the application based on the selected mode
    if args.mode == "standalone":
        app = create_standalone_app()
        logger.info("Created standalone Flask application")
    else:
        app = integrate_with_existing_app()
        logger.info("Integrated with existing Flask application")
    
    # Apply patches to override existing routes if requested
    if args.override:
        patch_existing_routes(app, override=True)
    
    # Add test page
    add_test_page(app)
    
    # Run the application
    logger.info(f"Starting server on port {args.port} (debug={args.debug})")
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)

if __name__ == "__main__":
    main() 