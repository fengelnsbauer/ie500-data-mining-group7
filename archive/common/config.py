# common/config.py

PIT_STOP_DURATION = 25000  # Pit stop penalty in milliseconds

TIRE_COMPOUND_EFFECTS = {
    3: {'base_speed': 0.98, 'degradation_per_lap': 500},   # Soft
    2: {'base_speed': 0.99, 'degradation_per_lap': 300},   # Medium  
    1: {'base_speed': 1.0, 'degradation_per_lap': 200},    # Hard
    4: {'base_speed': 1.05, 'degradation_per_lap': 200},   # Intermediate
    5: {'base_speed': 1.1, 'degradation_per_lap': 200},    # Wet
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
