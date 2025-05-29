# Deep RL Integration Guide

This guide explains how to integrate the Deep Reinforcement Learning (Deep RL) components into the main maritime routing system.

## Prerequisites

1. Install the required dependencies:
   ```bash
   pip install -r deep_rl_requirements.txt
   ```

2. Ensure you have access to:
   - Weather data sources (or the sample data provided)
   - Ship performance models
   - Grid-based representations of maritime routes

## Integration Steps

### 1. Add the Deep RL Router to the algorithms directory

```python
# Place the Deep RL router implementation in algorithms/ml/deep_rl_router.py
```

### 2. Update the AI Routing Integration

Modify the `algorithms/ai_routing_integration.py` file to include the Deep RL router:

```python
# Import the Deep RL router
from algorithms.ml.deep_rl_router import DeepRLRouter

# Add function to use Deep RL for route optimization
def deep_rl_route_optimization(start, end, env, ship, config, departure_time=None):
    """
    Calculate an optimized route using Deep Reinforcement Learning.
    
    Args:
        start (tuple): Start coordinates (lat, lon)
        end (tuple): End coordinates (lat, lon)
        env (Environment): Environment model with weather data
        ship (Ship): Ship model with performance characteristics
        config (dict): Configuration parameters
        departure_time (datetime): Departure time
        
    Returns:
        dict: Optimized route with waypoints and metadata
    """
    try:
        # Initialize the Deep RL router
        router = DeepRLRouter(
            env_model=env,
            ship_model=ship,
            config=config.get('deep_rl', {})
        )
        
        # Calculate the route
        route = router.calculate_route(start, end, departure_time)
        
        return route
    except Exception as e:
        logger.error(f"Deep RL route calculation failed: {e}")
        # Fall back to traditional routing method
        return ai_enhanced_route_optimization(start, end, env, ship, config, departure_time)

# Add function to generate multiple routes
def generate_deep_rl_routes(start, end, env, ship, config, departure_time=None, num_routes=3):
    """
    Generate multiple route alternatives using Deep RL with different parameters.
    
    Args:
        start (tuple): Start coordinates (lat, lon)
        end (tuple): End coordinates (lat, lon)
        env (Environment): Environment model with weather data
        ship (Ship): Ship model with performance characteristics
        config (dict): Configuration parameters
        departure_time (datetime): Departure time
        num_routes (int): Number of route alternatives to generate
        
    Returns:
        list: List of route dictionaries with waypoints and metadata
    """
    routes = []
    
    try:
        # Generate primary route
        primary_route = deep_rl_route_optimization(
            start=start,
            end=end,
            env=env,
            ship=ship,
            config=config,
            departure_time=departure_time
        )
        
        routes.append(primary_route)
        
        # Generate alternative routes with different optimization parameters
        if num_routes > 1:
            # Copy config for modifications
            fuel_config = copy.deepcopy(config)
            fuel_config['deep_rl']['optimization_weights'] = {
                'fuel': 0.7,
                'time': 0.1,
                'safety': 0.2
            }
            
            speed_config = copy.deepcopy(config)
            speed_config['deep_rl']['optimization_weights'] = {
                'fuel': 0.1,
                'time': 0.8,
                'safety': 0.1
            }
            
            safety_config = copy.deepcopy(config)
            safety_config['deep_rl']['optimization_weights'] = {
                'fuel': 0.1,
                'time': 0.1,
                'safety': 0.8
            }
            
            # Generate fuel-efficient route
            fuel_route = deep_rl_route_optimization(
                start=start,
                end=end,
                env=env,
                ship=ship,
                config=fuel_config,
                departure_time=departure_time
            )
            fuel_route['route_type'] = 'fuel_optimized'
            routes.append(fuel_route)
            
            # Generate speed-optimized route
            speed_route = deep_rl_route_optimization(
                start=start,
                end=end,
                env=env,
                ship=ship,
                config=speed_config,
                departure_time=departure_time
            )
            speed_route['route_type'] = 'speed_optimized'
            routes.append(speed_route)
            
            # Generate safety-optimized route if more routes requested
            if num_routes > 3:
                safety_route = deep_rl_route_optimization(
                    start=start,
                    end=end,
                    env=env,
                    ship=ship,
                    config=safety_config,
                    departure_time=departure_time
                )
                safety_route['route_type'] = 'safety_optimized'
                routes.append(safety_route)
        
        return routes[:num_routes]  # Limit to requested number
        
    except Exception as e:
        logger.error(f"Multiple Deep RL route generation failed: {e}")
        # Fall back to traditional routing
        return [ai_enhanced_route_optimization(start, end, env, ship, config, departure_time)]
```

### 3. Add API Endpoint

Update `app.py` to add an endpoint for the Deep RL router:

```python
@app.route('/api/routes/deep_rl', methods=['POST'])
def calculate_deep_rl_routes():
    """Calculate multiple alternative routes between two points using Deep RL approach."""
    data = request.json
    
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400
    
    # Extract parameters from request
    start = data.get('start')
    end = data.get('end')
    ship_type = data.get('ship_type', 'container_medium')
    departure_time = data.get('departure_time')
    num_routes = data.get('num_routes', 3)
    
    # Validate required parameters
    if not start or not end:
        return jsonify({'success': False, 'error': 'Start and end points are required'}), 400
    
    # Parse departure time if provided
    if departure_time:
        try:
            departure_time = datetime.strptime(departure_time, '%Y-%m-%dT%H:%M:%S')
        except ValueError:
            departure_time = datetime.now()
    else:
        departure_time = datetime.now()
    
    try:
        # Initialize environment and ship models
        env_model = Environment(config=config['environment'])
        ship_model = Ship(ship_type=ship_type, config=config['ship'])
        
        # Calculate multiple routes
        routes = generate_deep_rl_routes(
            start=start,
            end=end,
            env=env_model,
            ship=ship_model,
            config=config,
            departure_time=departure_time,
            num_routes=num_routes
        )
        
        # Format response
        result = {
            'success': True,
            'count': len(routes),
            'routes': routes
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error calculating Deep RL routes: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
```

### 4. Add Configuration

Add Deep RL configuration to your config files:

```yaml
# In config/config.yaml or similar
deep_rl:
  model:
    learning_rate: 0.001
    discount_factor: 0.95
    hidden_layer_sizes: [128, 64]
    activation: "relu"
    
  training:
    episodes: 1000
    batch_size: 64
    update_frequency: 10
    
  optimization_weights:
    fuel: 0.33
    time: 0.33
    safety: 0.34
    
  grid:
    resolution: 0.1  # degrees
    max_distance: 100  # nautical miles
```

### 5. Add Frontend Support

Add UI components to request and display Deep RL routes:

```javascript
// In your frontend code
async function requestDeepRLRoutes() {
  const response = await fetch('/api/routes/deep_rl', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      start: [startLat, startLon],
      end: [endLat, endLon],
      ship_type: selectedShipType,
      departure_time: departureTime.toISOString(),
      num_routes: 3
    })
  });
  
  const data = await response.json();
  
  if (data.success) {
    // Display routes on the map
    displayMultipleRoutes(data.routes);
  } else {
    showError(data.error);
  }
}

function displayMultipleRoutes(routes) {
  // Clear existing routes
  clearRoutes();
  
  // Define colors for different route types
  const colors = {
    'primary': '#0066FF',
    'fuel_optimized': '#00CC00',
    'speed_optimized': '#FF3300',
    'safety_optimized': '#9900CC'
  };
  
  // Add each route to the map
  routes.forEach((route, index) => {
    const routeType = route.route_type || 'primary';
    const color = colors[routeType] || colors.primary;
    
    addRouteToMap(route.waypoints, color, `Route ${index + 1}: ${routeType}`);
    
    // Add route metrics to sidebar
    addRouteMetricsToSidebar(route, index);
  });
}
```

## Testing the Integration

1. Test the Deep RL Router component in isolation:
   ```bash
   python -m tests.test_deep_rl_router
   ```

2. Test the API endpoint:
   ```bash
   curl -X POST http://localhost:5000/api/routes/deep_rl \
     -H "Content-Type: application/json" \
     -d '{"start": [8.4, 76.9], "end": [9.9, 78.1], "ship_type": "container_medium", "num_routes": 3}'
   ```

3. Verify the frontend integration by opening the web interface and testing the route generation.

## Performance Optimization

If you encounter performance issues with the Deep RL implementation:

1. Use TensorFlow's performance optimizations:
   ```python
   import tensorflow as tf
   tf.config.optimizer.set_jit(True)  # Enable XLA compilation
   ```

2. Pre-compute grid-based representations of the environment to reduce computational overhead.

3. Use lower grid resolution for initial route planning, then refine with higher resolution.

4. Implement model serving for production deployments using TensorFlow Serving or similar tools.

## Troubleshooting

- **Memory Issues**: Reduce batch size or model complexity
- **Slow Route Generation**: Use cached weather data and pre-trained models
- **Inconsistent Routes**: Check for randomness in the algorithm or adjust exploration parameters
- **Integration Errors**: Verify data formats and coordinate transformations between system components

## Further Resources

- Deep RL Architecture Documentation: See `DEEP_RL_ARCHITECTURE.txt`
- Full Technical Description: See `DEEP_RL_MARITIME_ROUTING.md`
- Demo Implementation: See `run_deep_rl_demo.py` 