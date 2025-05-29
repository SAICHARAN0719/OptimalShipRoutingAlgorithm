"""
Fallback classes for Environment and Ship to ensure the application runs
even when main module imports fail.
"""

import os
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class Ship:
    """Fallback Ship class for when the import fails."""
    def __init__(self, length=200.0, weight=50000.0, fuel_rate=0.3, beam=None, draft=None, 
                 max_speed=25.0, current_fuel=None, ship_type=None, config=None):
        self.ship_type = ship_type or 'container_medium'
        self.length = length
        self.weight = weight
        self.fuel_rate = fuel_rate
        self.beam = beam
        self.draft = draft
        self.max_speed = max_speed
        self.current_fuel = current_fuel
        self.cruising_speed = max_speed * 0.8
        self.fuel_capacity = 2000  # tons
        self.fuel_consumption_rate = 0.1  # tons per nautical mile
        self.max_wave_height = 5.0
        self.max_wind_speed = 40.0
        self.service_speed = max_speed * 0.8
        self.specific_fuel_consumption = 180
        self.max_power = 15000
        
        # If config is provided, override defaults
        if config:
            self.max_speed = config.get('max_speed', self.max_speed)
            self.cruising_speed = config.get('cruising_speed', self.cruising_speed)
            self.fuel_capacity = config.get('fuel_capacity', self.fuel_capacity)
            self.fuel_consumption_rate = config.get('fuel_consumption_rate', self.fuel_consumption_rate)
            self.length = config.get('length', self.length)
            self.weight = config.get('weight', self.weight)
            self.fuel_rate = config.get('fuel_rate', self.fuel_rate)
    
    def get_parameters(self):
        return {
            'type': self.ship_type,
            'max_speed_knots': self.max_speed,
            'cruising_speed_knots': self.cruising_speed,
            'fuel_capacity_tons': self.fuel_capacity,
            'fuel_consumption_rate': self.fuel_consumption_rate
        }

class Environment:
    """Fallback Environment class for when the import fails."""
    def __init__(self, data_path=None, config=None):
        self.data_path = data_path
        self.config = config or {}
        
        # If config is provided, extract data_path
        if config and not data_path:
            self.data_path = config.get('data_path', 'data/weather_data.json')
        
        logger.info(f"Initialized fallback Environment with data_path: {self.data_path}")
    
    def get_bounds(self):
        """Get geographical bounds of available environmental data."""
        return {
            'latitude': (0, 90),
            'longitude': (-180, 180),
            'time': (datetime.now() - timedelta(days=30), datetime.now() + timedelta(days=30))
        }
    
    def get_conditions(self, position, time=None):
        """Get environmental conditions at a specific position and time."""
        # Return default conditions
        return {
            'wave_height': 1.0,  # meters
            'wind_speed': 10.0,  # knots
            'wind_direction': 0.0,  # degrees
            'current_speed': 0.5,  # knots
            'current_direction': 0.0,  # degrees
            'visibility': 10.0,  # nautical miles
            'is_safe': True
        }
    
    def is_safe_position(self, position):
        """Check if a position is safe for navigation."""
        # Default implementation always returns True
        return True
    
    def is_safe_conditions(self, conditions):
        """Check if environmental conditions are safe for navigation."""
        # Basic safety check
        return (conditions.get('wave_height', 0) < 4.0 and 
                conditions.get('wind_speed', 0) < 30.0) 