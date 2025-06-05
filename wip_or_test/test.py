angle = 135.0
radius = 88.0

radius_factor = min(1.0, 100.0 / max(radius, 1.0))  # Normalisieren: Kleiner Radius -> hoher Wert
angle_factor = min(1.0, abs(angle) / 90.0)  # Normalisieren: Größerer Winkel -> höherer Wert

print(radius_factor * angle_factor)