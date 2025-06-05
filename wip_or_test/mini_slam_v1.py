import numpy as np


def optimiere_trajektorie(mittelpunkte, streckenbreite, current_segment=None, next_segment=None,
                          distance_to_next_segment=None, ):
    """
    Optimiert die Trajektorie basierend auf
    Informationen über aktuelle und kommende Streckensegmente.

    Args:
        mittelpunkte: Liste von (x, y) Koordinaten der Streckenmittelpunkte
        streckenbreite: Breite der Strecke in Metern
        current_segment: Dictionary mit Informationen zum aktuellen Streckensegment
        next_segment: Dictionary mit Informationen zum nächsten Streckensegment
        distance_to_next_segment: Distanz zum nächsten Segment in Metern
        prev_offset: Vorheriger Offset-Wert für sanfte Übergänge

    Returns:
        Tupel aus: optimierte Trajektorie, aktuelle Geschwindigkeit, aktueller Offset
    """
    # Maximale sichere Verschiebung (65% der halben Streckenbreite)
    max_verschiebung = (streckenbreite / 2) * 0.65

    # Mindestanzahl von Punkten prüfen
    if len(mittelpunkte) < 3:
        return mittelpunkte.copy()

    # Standardberechnungen für Krümmungen und Normalvektoren beibehalten
    kruemmungen = berechne_kruemmungen(mittelpunkte)
    normalvektoren = berechne_normalvektoren(mittelpunkte)

    soll_verschiebung = 0

    # Wenn Segmentinformationen verfügbar sind, nutze diese für bessere Entscheidungen
    if current_segment is not None and next_segment is not None and distance_to_next_segment is not None:
        soll_verschiebung = berechne_segmentbasierte_verschiebung(
            current_segment, next_segment, distance_to_next_segment, max_verschiebung)

    # Verschiebung auf alle Punkte anwenden
    racing_line = []
    for i in range(len(mittelpunkte)):
        x, y, z = mittelpunkte[i]
        nx, ny = normalvektoren[i]
        neuer_punkt = (x + nx * soll_verschiebung, y + ny * soll_verschiebung)
        racing_line.append(neuer_punkt)

    # Geschwindigkeit basierend auf optimierter Trajektorie und Segmentinformationen berechnen
    current_speed = speeds_for_racing_line(racing_line, current_segment, next_segment, distance_to_next_segment)

    return racing_line, current_speed, soll_verschiebung


def berechne_segmentbasierte_verschiebung(current_segment, next_segment, distance_to_next_segment, max_verschiebung, last_offset_previous_segment):
    """
    Berechnet die optimale Verschiebung basierend auf Segmentinformationen.

    Args:
        current_segment: Dictionary mit Informationen zum aktuellen Segment
        next_segment: Dictionary mit Informationen zum nächsten Segment
        distance_to_next_segment: Distanz zum nächsten Segment
        max_verschiebung: Maximale erlaubte Verschiebung

    Returns:
        Optimale Verschiebung (-: nach rechts, +: nach links)
    """
    # Segment-Typen basierend auf Radius bestimmen
    # 0 = Gerade, > 0 = Linkskurve, < 0 = Rechtskurve
    current_type = get_segment_type(current_segment)
    next_type = get_segment_type(next_segment)

    # Übergangszone definieren - wann mit der Vorbereitung auf das nächste Segment beginnen
    distance_percentage_driven_in_this_segment = distance_to_next_segment / current_segment['distance']

    # Fall 1: Wir sind auf einer Geraden
    if current_type == "STRAIGHT":
        if next_type == "RIGHT_CURVE":
            wanted_offset_at_end_of_segment = -0.8
        else:
            wanted_offset_at_end_of_segment = 0.8
        delta_offset = wanted_offset_at_end_of_segment - last_offset_previous_segment
        current_wanted_offset = delta_offset * distance_percentage_driven_in_this_segment + last_offset_previous_segment

    # Fall 2: Wir sind in einer Linkskurve
    elif current_type == "LEFT_CURVE":
        if next_type == "STRAIGHT":
            wanted_offset_apex = -0.8
            wanted_offset_at_end_of_segment = 0.8
        else: # RIGHT CURVE
            wanted_offset_apex = 0
            wanted_offset_at_end_of_segment = -0.8
        if wanted_offset_apex != 0:
            if distance_percentage_driven_in_this_segment < 0.5:
                delta_offset = wanted_offset_apex - last_offset_previous_segment
                current_wanted_offset = delta_offset * distance_percentage_driven_in_this_segment * 2 + last_offset_previous_segment
            else:
                delta_offset = wanted_offset_at_end_of_segment - last_offset_previous_segment
                current_wanted_offset = delta_offset * distance_percentage_driven_in_this_segment + last_offset_previous_segment
        else:
            delta_offset = wanted_offset_at_end_of_segment - last_offset_previous_segment
            current_wanted_offset = delta_offset * distance_percentage_driven_in_this_segment + last_offset_previous_segment

    # Fall 3: Wir sind in einer Rechtskurve
    else:  # RIGHT_CURVE
        if next_type == "STRAIGHT":
            wanted_offset_apex = 0.8
            wanted_offset_at_end_of_segment = -0.8
        else: # RIGHT CURVE
            wanted_offset_apex = 0
            wanted_offset_at_end_of_segment = 0.8
        if wanted_offset_apex != 0:
            if distance_percentage_driven_in_this_segment < 0.5:
                delta_offset = wanted_offset_apex - last_offset_previous_segment
                current_wanted_offset = delta_offset * distance_percentage_driven_in_this_segment * 2 + last_offset_previous_segment
            else:
                delta_offset = wanted_offset_at_end_of_segment - last_offset_previous_segment
                current_wanted_offset = delta_offset * distance_percentage_driven_in_this_segment + last_offset_previous_segment
        else:
            delta_offset = wanted_offset_at_end_of_segment - last_offset_previous_segment
            current_wanted_offset = delta_offset * distance_percentage_driven_in_this_segment + last_offset_previous_segment

    return current_wanted_offset


def get_segment_type(segment):
    """Bestimmt den Typ des Segments basierend auf Radius/Winkel."""
    if segment is None:
        return "STRAIGHT"

    if 'Radius' not in segment or segment['Radius'] == 0 or segment['Angle'] == 0:
        return "STRAIGHT"
    elif segment['Angle'] > 0:
        return "LEFT_CURVE"
    else:
        return "RIGHT_CURVE"



def speeds_for_racing_line(racing_line, current_segment=None, next_segment=None,
                           distance_to_next_segment=None, max_speed=3.0, min_speed=0.5):
    """
    Berechnet die optimale Geschwindigkeit basierend auf der Racing Line und Segmentinformationen.

    Args:
        racing_line: Liste von (x, y) Koordinaten der Racing Line
        current_segment: Dictionary mit Informationen zum aktuellen Segment
        next_segment: Dictionary mit Informationen zum nächsten Segment
        distance_to_next_segment: Distanz zum nächsten Segment
        max_speed: Maximale Geschwindigkeit in m/s
        min_speed: Minimale Geschwindigkeit in m/s

    Returns:
        Optimale Geschwindigkeit in m/s
    """
    # Standardberechnung basierend auf Winkeln der Trajektorie
    angles = calculate_segment_angles(racing_line)
    mean_angle = np.mean(angles) if angles else 0

    # Basisgeschwindigkeit aus Winkeln
    angle_speed_factor = max(0.0, 1 - (mean_angle / 45.0) * 0.8)
    base_speed = min_speed + (max_speed - min_speed) * angle_speed_factor

    # Wenn keine Segmentinformationen verfügbar sind, nutze nur die Basisgeschwindigkeit
    if not current_segment or not next_segment:
        return base_speed

    # Sonst: Erweiterte Geschwindigkeitsberechnung mit Segmentinformationen
    current_type = get_segment_type(current_segment)
    next_type = get_segment_type(next_segment)

    # Optimale Geschwindigkeit für aktuelles Segment
    if current_type == "STRAIGHT":
        current_segment_speed = max_speed
    else:
        # Für Kurven: Je kleiner der Radius, desto langsamer
        curve_severity = calculate_curve_severity(current_segment)
        current_segment_speed = min_speed + (max_speed - min_speed) * (1 - curve_severity * 0.9)

    # Optimale Geschwindigkeit für nächstes Segment
    if next_type == "STRAIGHT":
        next_segment_speed = max_speed
    else:
        curve_severity = calculate_curve_severity(next_segment)
        next_segment_speed = min_speed + (max_speed - min_speed) * (1 - curve_severity * 0.9)

    # Abbremszone definieren - wann mit dem Abbremsen auf die nächste Segmentgeschwindigkeit beginnen
    braking_distance = (current_segment_speed - next_segment_speed) * 1.2  # Einfache Schätzung

    # Wenn wir in der Abbremszone sind und das nächste Segment langsamer ist
    if distance_to_next_segment < braking_distance and next_segment_speed < current_segment_speed:
        # Lineares Abbremsen
        braking_factor = distance_to_next_segment / braking_distance
        target_speed = next_segment_speed + (current_segment_speed - next_segment_speed) * braking_factor
    else:
        target_speed = current_segment_speed

    # Kombination mit der aus Winkeln berechneten Geschwindigkeit
    # (gewichtet, da die direkte Messung der Winkel in der Trajektorie wichtig bleibt)
    final_speed = 0.7 * target_speed + 0.3 * base_speed

    return max(min_speed, min(max_speed, final_speed))


def berechne_kruemmungen(punkte):
    """Berechnet die Krümmung für jeden Punkt der Trajektorie."""
    n = len(punkte)
    kruemmungen = [0.0] * n

    for i in range(1, n - 1):
        v1 = (punkte[i][0] - punkte[i - 1][0], punkte[i][1] - punkte[i - 1][1])
        v2 = (punkte[i + 1][0] - punkte[i][0], punkte[i + 1][1] - punkte[i][1])

        kreuzprodukt = v1[0] * v2[1] - v1[1] * v2[0]
        len_v1 = max(0.0001, (v1[0] ** 2 + v1[1] ** 2) ** 0.5)
        len_v2 = max(0.0001, (v2[0] ** 2 + v2[1] ** 2) ** 0.5)

        kruemmungen[i] = kreuzprodukt / (len_v1 * len_v2)

    if n > 2:
        kruemmungen[0] = kruemmungen[1]
        kruemmungen[n - 1] = kruemmungen[n - 2]

    return kruemmungen


def berechne_normalvektoren(punkte):
    """Berechnet die Normalvektoren (senkrecht zur Fahrtrichtung) für jeden Punkt."""
    n = len(punkte)
    normalvektoren = [(0.0, 0.0)] * n

    # Ersten Punkt behandeln
    if n > 1:
        dx = punkte[1][0] - punkte[0][0]
        dy = punkte[1][1] - punkte[0][1]
        laenge = max(0.0001, (dx ** 2 + dy ** 2) ** 0.5)
        normalvektoren[0] = (-dy / laenge, dx / laenge)

    # Mittlere Punkte
    for i in range(1, n - 1):
        dx = punkte[i + 1][0] - punkte[i - 1][0]
        dy = punkte[i + 1][1] - punkte[i - 1][1]
        laenge = max(0.0001, (dx ** 2 + dy ** 2) ** 0.5)
        normalvektoren[i] = (-dy / laenge, dx / laenge)

    # Letzten Punkt behandeln
    if n > 1:
        dx = punkte[n - 1][0] - punkte[n - 2][0]
        dy = punkte[n - 1][1] - punkte[n - 2][1]
        laenge = max(0.0001, (dx ** 2 + dy ** 2) ** 0.5)
        normalvektoren[n - 1] = (-dy / laenge, dx / laenge)

    return normalvektoren


def berechne_vorausschau(kruemmungen):
    """Erkennt zukünftige Kurven basierend auf Krümmungswerten."""
    n = len(kruemmungen)
    vorausschau = [0.0] * n

    for i in range(n):
        max_kruemmung = 0.0
        for j in range(i + 1, min(i + 6, n)):
            if abs(kruemmungen[j]) > abs(max_kruemmung):
                max_kruemmung = kruemmungen[j]
        vorausschau[i] = max_kruemmung

    return vorausschau


def calculate_segment_angles(points):
    """Berechnet die Winkel zwischen aufeinanderfolgenden Segmenten."""
    points = np.array(points)
    if len(points) < 3:
        return []

    vectors = points[1:] - points[:-1]
    angles = []

    for i in range(len(vectors) - 1):
        v1 = vectors[i]
        v2 = vectors[i + 1]

        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            angles.append(0)
            continue

        cos_angle = dot_product / (norm_v1 * norm_v2)
        cos_angle = max(min(cos_angle, 1.0), -1.0)

        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        cross_product = np.cross(np.append(v1, 0), np.append(v2, 0))[2]
        if cross_product < 0:
            angle_deg = 360 - angle_deg

        if angle_deg > 180:
            angle_deg = 360 - angle_deg

        angles.append(angle_deg)

    return angles