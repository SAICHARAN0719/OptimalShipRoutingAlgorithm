#!/usr/bin/env python
"""
Analyze route variations generated for Visakhapatnam to Shanghai
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load the route data
    with open('visakhapatnam_shanghai_variations.json', 'r') as f:
        data = json.load(f)
    
    # Print summary for each route
    print("\n===== ROUTE VARIATIONS COMPARISON =====")
    # Get timestamp from the first route if overall timestamp isn't available
    timestamp = data.get('timestamp', data['routes'][0].get('timestamp', 'Unknown'))
    print(f"From Visakhapatnam to Shanghai - Timestamp: {timestamp}")
    print("=" * 40)
    
    for i, route in enumerate(data['routes']):
        metrics = route['metrics']
        print(f"Route {i+1} ({route['route_type']})")
        print(f"  Distance: {metrics['distance']:.1f} nautical miles")
        print(f"  Duration: {metrics['duration']:.1f} hours ({metrics['duration']/24:.1f} days)")
        print(f"  Fuel consumption: {metrics['fuel_consumption']:.1f} tons")
        print(f"  CO2 emissions: {metrics['co2_emissions']:.1f} tons")
        print(f"  Waypoints: {len(route['waypoints'])//2}")
        print("-" * 40)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Fill background with light blue for ocean
    plt.fill_between([-180, 180], [-90, -90], [90, 90], color='lightblue', zorder=0)
    
    # Define colors for routes
    colors = ['blue', 'green', 'red']
    
    # Plot each route
    for i, route in enumerate(data['routes']):
        # Reshape waypoints into pairs
        waypoints = np.array(route['waypoints']).reshape(-1, 2)
        
        # Plot the route
        plt.plot(
            waypoints[:, 1], waypoints[:, 0], 
            '-', 
            color=colors[i], 
            linewidth=2.5, 
            label=f"Route {i+1} - {route['metrics']['distance']:.1f} nm"
        )
    
    # Mark start and end points
    start = data['routes'][0]['start']
    end = data['routes'][0]['end']
    plt.plot(start[1], start[0], 'ko', markersize=10, label='Visakhapatnam')
    plt.plot(end[1], end[0], 'kx', markersize=10, label='Shanghai')
    
    # Add title and labels
    plt.title('Route Variations: Visakhapatnam to Shanghai', fontsize=16)
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    
    # Save the figure
    plt.savefig('route_comparison_analysis.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as route_comparison_analysis.png")
    print("=" * 40)

if __name__ == "__main__":
    main() 