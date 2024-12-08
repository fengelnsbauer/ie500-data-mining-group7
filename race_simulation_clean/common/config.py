# config.py
PIT_STOP_DURATION = 20000  # ms

TIRE_COMPOUND_EFFECTS = {
    3: {'base_speed': 0.98, 'degradation_per_lap': 500},   # Soft
    2: {'base_speed': 0.99, 'degradation_per_lap': 300},   # Medium  
    1: {'base_speed': 1.0, 'degradation_per_lap': 200},    # Hard
}

DEFAULT_WEATHER = {
            'TrackTemp': 35.0,
            'AirTemp': 25.0,
            'Humidity': 50.0
        }

TEAM_COLORS = {
    'alpine': '#fe86bc',
    'aston martin': '#006f62',
    'ferrari': '#dc0000',
    'haas': '#B6BABD',
    'mclaren': '#ff8700',
    'mercedes': '#27F4D2',
    'red bull': '#3671C6',
    'sauber': '#52E252',
    'williams': '#64C4FF',
    'rb': '#6692FF',
    'unknown': '#CCCCCC',  # Default color for unknown teams
}