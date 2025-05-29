# Enhanced Maritime Routing System

This enhancement adds robust land avoidance capabilities to the maritime routing application, especially focusing on ensuring all routes (including multiple route variations) avoid land masses.

## Key Features

- **Improved Land Detection**: Enhanced detection of land masses, particularly focusing on challenging areas like the Indian subcontinent and Southeast Asia
- **Multiple Route Variations**: Generate multiple alternative routes that all avoid land collisions
- **Strategic Waypoint Placement**: Automatically places waypoints to navigate around coastlines and through key shipping lanes
- **Optimized Waypoints**: Removes unnecessary waypoints while maintaining safe navigation paths
- **Compatible API**: Integrates with the existing application through a compatible API layer

## Problem Solved

The original routing system sometimes generated routes that crossed land areas when multiple routes were requested. This enhancement ensures that all routes, even when generating multiple variations, properly avoid land masses.

### Specific Improvements:

1. Routes from Visakhapatnam to Shanghai and other routes crossing the Indian peninsula now navigate through proper shipping lanes
2. All route variations maintain a safe distance from coastlines
3. Routes between mainland and islands properly navigate through sea channels

## Installation and Setup

### Prerequisites

- Python 3.6+
- Flask
- NumPy
- Matplotlib (for visualization)

### Installation Steps

1. Clone the repository or ensure you have the updated project files
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Ensure the following files are in your project directory:
   - `route_variations.py`: Route variation generator
   - `routing_integration.py`: Integration with main application
   - `api_integration.py`: API layer for Flask integration
   - `deploy_enhanced_routing.py`: Deployment script

## Usage

### Running the Enhanced Routing Server

You can run the enhanced routing system in two modes:

#### Standalone Mode

```bash
python deploy_enhanced_routing.py --mode standalone --port 5000
```

This starts a new Flask server with only the enhanced routing functionality.

#### Integrated Mode (Default)

```bash
python deploy_enhanced_routing.py
```

This integrates the enhanced routing with your existing application.

#### Additional Options

- `--debug`: Run in debug mode
- `--port <number>`: Specify the port (default: 5000)
- `--override`: Override existing API endpoints with enhanced versions

### Using the API

#### Calculate a Single Route

```http
POST /api/enhanced-route
Content-Type: application/json

{
  "start": [17.6868, 83.2185],
  "end": [31.2304, 121.4737],
  "ship_type": "container_medium",
  "route_type": "standard"
}
```

#### Calculate Multiple Routes

```http
POST /api/enhanced-multiple-routes
Content-Type: application/json

{
  "start": [17.6868, 83.2185],
  "end": [31.2304, 121.4737],
  "ship_type": "container_medium",
  "num_routes": 3
}
```

#### Check if a Route Crosses Land

```http
POST /api/check-land-crossing
Content-Type: application/json

{
  "waypoints": [
    [17.6868, 83.2185],
    [20.0, 90.0],
    [25.0, 100.0],
    [31.2304, 121.4737]
  ]
}
```

### Test Page

For a visual testing interface, navigate to:

```
http://localhost:5000/test-routing
```

This page allows you to enter coordinates, select ship types, and visualize the generated routes on an interactive map.

## Testing

To test the enhanced routing system:

```bash
python test_visakhapatnam_shanghai_enhanced.py
```

This will calculate and visualize routes from Visakhapatnam to Shanghai and save the results to JSON files.

## Implementation Details

### Route Variation Types

The system supports several types of route variations:

1. **Strategic Routes**: Uses geographical knowledge to route through known shipping lanes
2. **Waypoint Variations**: Modifies existing routes with small deviations
3. **Direct Variations**: Creates new routes with smooth curves

### Land Detection Methods

The system uses multiple methods to detect land:

1. Primary method: Uses the `TerrainRecognition` class from `dynamic_routing.py` when available
2. Fallback method: Uses a simplified geographical detection system built into `routing_integration.py`

### Caching

For performance optimization, calculated routes are cached in memory and on disk in the `route_cache` directory.

## Customization

### Adding New Ship Types

To add new ship types, modify the `_get_ship_parameters` method in `routing_integration.py`.

### Adding Known Locations

To add more known locations for naming, modify the `_get_location_name` method in `routing_integration.py`.

## Troubleshooting

### Common Issues

1. **API Errors**: Ensure the server is running and check the log files
2. **Missing Routes**: Check if the land detection is working correctly by using the `/api/check-land-crossing` endpoint
3. **Performance Issues**: Routes are cached for faster performance; clear the `route_cache` directory if needed

## Contributing

Contributions to improve the routing system are welcome! Please ensure that all changes maintain the core functionality of avoiding land masses.

## License

This project is licensed under the same license as the main maritime routing application. 