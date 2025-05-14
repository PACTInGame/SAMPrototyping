import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
import matplotlib.colors as mcolors


def optimiere_trajektorie(mittelpunkte, streckenbreite, prev_offset=0):
    """
    Optimiert die Trajektorie für autonomes Fahren nach Racing-Line-Prinzip:
    - Vor Kurven: Nach außen fahren
    - In Kurven: Nach innen fahren
    - In diser Version sanftere Übergänge, durch merken des letzen Offsets (andere Logik zum Verschieben)
    Args:
        mittelpunkte: Liste von (x, y) Koordinaten der Streckenmittelpunkte mind. 3
        streckenbreite: Breite der Strecke in Metern

    Returns:
        Liste mit optimierten (x, y) Koordinaten
    """
    # Maximale sichere Verschiebung (65% der halben Streckenbreite)
    max_verschiebung = (streckenbreite / 2) * 0.65

    # Mindestanzahl von Punkten prüfen
    # TODO achtung, nicht in C++ übernehmen, wenn points > 3 muss trotzdem speed berechnet werden (wrsl. min speed), und mittelpunkte den last offset nehmen.
    if len(mittelpunkte) < 3:
        return mittelpunkte.copy()

    # 1. Krümmung für jeden Punkt berechnen
    def berechne_kruemmungen(punkte):
        n = len(punkte)
        kruemmungen = [0.0] * n

        for i in range(1, n - 1):
            # Vektoren zwischen Punkten
            v1 = (punkte[i][0] - punkte[i - 1][0], punkte[i][1] - punkte[i - 1][1])
            v2 = (punkte[i + 1][0] - punkte[i][0], punkte[i + 1][1] - punkte[i][1])

            # Kreuzprodukt berechnen (bestimmt Drehrichtung)
            kreuzprodukt = v1[0] * v2[1] - v1[1] * v2[0]

            # Normierung
            len_v1 = max(0.0001, (v1[0] ** 2 + v1[1] ** 2) ** 0.5)
            len_v2 = max(0.0001, (v2[0] ** 2 + v2[1] ** 2) ** 0.5)

            # Krümmung: positiv = Linkskurve, negativ = Rechtskurve
            kruemmungen[i] = kreuzprodukt / (len_v1 * len_v2)

        # Randwerte setzen
        if n > 2:
            kruemmungen[0] = kruemmungen[1]
            kruemmungen[n - 1] = kruemmungen[n - 2]

        return kruemmungen

    # 2. Normalvektoren (senkrecht zur Fahrtrichtung) berechnen
    def berechne_normalvektoren(punkte):
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
            # Richtung vom vorherigen zum nächsten Punkt
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

    # 3. Vorausschau - Zukünftige Kurven erkennen
    def berechne_vorausschau(kruemmungen):
        n = len(kruemmungen)
        vorausschau = [0.0] * n

        for i in range(n):
            # Maximale Krümmung in den nächsten Punkten finden
            max_kruemmung = 0.0
            richtung = 0

            # Bis zu 5 Punkte vorausschauen (oder weniger, falls nicht verfügbar)
            for j in range(i + 1, min(i + 6, n)):
                if abs(kruemmungen[j]) > abs(max_kruemmung):
                    max_kruemmung = kruemmungen[j]
                    richtung = 1 if kruemmungen[j] > 0 else -1

            # Speichern der vorausschauenden Krümmung
            vorausschau[i] = max_kruemmung

        return vorausschau

    # Krümmungen berechnen
    kruemmungen = berechne_kruemmungen(mittelpunkte)

    # Normalvektoren berechnen
    normalvektoren = berechne_normalvektoren(mittelpunkte)

    # Vorausschau berechnen
    vorausschau = berechne_vorausschau(kruemmungen)
    print(vorausschau)
    # 4. Verschiebungen berechnen
    verschiebungen = [0.0] * len(mittelpunkte)

    # Parameter für die Erkennung
    kurven_schwellwert = 0.4  # Ab wann ist es eine Kurve?
    vorausschau_schwellwert = 0.1  # Ab wann vorbereiten auf Kurve?
    soll_verschiebung = 0
    # Priorität 1: Sind wir in einer Kurve?
    if abs(kruemmungen[0]) > kurven_schwellwert:
        # In der Kurve nach innen verschieben
        richtung = 1 if kruemmungen[0] > 0 else -1
        faktor = min(1.0, abs(kruemmungen[0]) * 3)
        soll_verschiebung = max_verschiebung * faktor * richtung


    # Priorität 2: Kommt eine Kurve?
    elif abs(vorausschau[0]) > vorausschau_schwellwert:
        # Vor der Kurve nach außen verschieben
        richtung = -1 if vorausschau[0] > 0 else 1  # Gegenteil der Kurvenrichtung
        faktor = min(1.0, abs(vorausschau[0]) * 5)
        soll_verschiebung = max_verschiebung * faktor * richtung
    print(prev_offset, soll_verschiebung)
    verschiebungen[0] = prev_offset * 0.75 + soll_verschiebung * 0.25
    verschiebungen = [verschiebungen[0] for _ in range(len(mittelpunkte))]

    # 6. Racing Line erzeugen
    racing_line = []

    for i in range(len(mittelpunkte)):
        x, y, z = mittelpunkte[i]
        nx, ny = normalvektoren[i]

        # Verschiebung anwenden
        neuer_punkt = (x + nx * verschiebungen[i], y + ny * verschiebungen[i])
        racing_line.append(neuer_punkt)
    current_speed = speeds_for_racing_line(racing_line)
    return racing_line, current_speed, verschiebungen[0]


def speeds_for_racing_line(racing_line, max_speed=3, min_speed=0.5):
    """
    Berechnet die Geschwindigkeiten entlang der Racing Line.

    Args:
        racing_line: Liste von (x, y) Koordinaten der Racing Line
        max_speed: Maximale Geschwindigkeit in m/s

    Returns:
        Int von Geschwindigkeiten für den aktuellen Punkt mit Blick auf die kommende Linie
    """

    def calculate_segment_angles(points):
        """
        Berechnet die Winkel zwischen aufeinanderfolgenden Segmenten einer Trajektorie

        Parameters:
        points (list or np.ndarray): Liste von Punkten [(x1,y1), (x2,y2), ..., (xn,yn)]

        Returns:
        list: Liste der Winkel in Grad zwischen aufeinanderfolgenden Segmenten
        """
        # Konvertiere Punkte zu numpy array, falls sie es noch nicht sind
        points = np.array(points)

        # Wir brauchen mindestens 3 Punkte für einen Winkel zwischen zwei Segmenten
        if len(points) < 3:
            return []

        # Berechne die Richtungsvektoren für jedes Segment
        vectors = points[1:] - points[:-1]

        angles = []

        # Berechne Winkel zwischen aufeinanderfolgenden Vektoren
        for i in range(len(vectors) - 1):
            v1 = vectors[i]
            v2 = vectors[i + 1]

            # Berechne den Winkel zwischen den Vektoren mittels Arcuscosinus des Skalarprodukts
            # normalisierter Vektoren
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            # Vermeidung von Division durch Null
            if norm_v1 == 0 or norm_v2 == 0:
                angles.append(0)
                continue

            cos_angle = dot_product / (norm_v1 * norm_v2)
            # Begrenze den Wert auf [-1, 1] um numerische Probleme zu vermeiden
            cos_angle = max(min(cos_angle, 1.0), -1.0)

            # Berechne den Winkel in Grad
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)

            # Bestimme die Richtung des Winkels (im oder gegen den Uhrzeigersinn)
            cross_product = np.cross(np.append(v1, 0), np.append(v2, 0))[2]
            if cross_product < 0:
                angle_deg = 360 - angle_deg

            angles.append(angle_deg)
        for i, angle in enumerate(angles):
            if angle > 180:
                angle = 360 - angle
                angles[i] = angle
        return angles

    speed = max_speed
    angles = calculate_segment_angles(racing_line)
    #for angle in angles:
    #    if angle > 44:
    #        angle = 44
    #    speed_factor = angle * -0.0222222 + 1
    #    new_speed = max(min_speed, max_speed * speed_factor)
    #    speed = min(new_speed, speed)

    mean_angle = np.mean(angles)
    speed_factor = mean_angle * -0.0222222 + 1
    speed = max(min_speed, max_speed * speed_factor)
    return speed


def load_coordinates_from_file(filename):
    """
    Loads coordinate data from a file and returns three lists of tuples.
    """
    with open(filename, "r") as file:
        lines = file.readlines()

    center, right, left = [], [], []
    current_section = None

    for line in lines:
        line = line.strip()
        if "Center Coordinates:" in line:
            current_section = center
        elif "Right Coordinates:" in line:
            current_section = right
        elif "Left Coordinates:" in line:
            current_section = left
        elif line and current_section is not None:
            current_section.append(tuple(map(float, line.strip("() ").split(','))))

    return center, right, left


def visualize_multiple_lines(arrays_of_points, labels=None, title="Multiple Lines Plot",
                             x_label="X-axis", y_label="Y-axis", colors=None,
                             line_styles=None, markers=None, figsize=(10, 6),
                             update_function=None):
    """
    Visualize multiple arrays of points as lines in an interactive plot.

    Parameters:
    -----------
    [... deine bisherigen Parameter ...]
    update_function : function, optional
        Function that returns updated data for dynamic plotting
    """

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Set default values for optional parameters
    n_lines = len(arrays_of_points)

    if labels is None:
        labels = [f"Line {i + 1}" for i in range(n_lines)]

    if colors is None:
        colors = [None] * n_lines

    if line_styles is None:
        styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
        line_styles = [styles[i % len(styles)] for i in range(n_lines)]

    if markers is None:
        marker_styles = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
        markers = [marker_styles[i % len(marker_styles)] for i in range(n_lines)]

    # Plot each line
    lines = []
    for i, points in enumerate(arrays_of_points):
        points_array = np.array(points)

        # Check if points are 2D (x, y) or just y values
        if points_array.ndim == 1:  # Only y values
            y = points_array
            x = np.arange(len(y))
        else:  # (x, y) pairs
            x = points_array[:, 0]
            y = points_array[:, 1]

        line, = ax.plot(x, y,
                        label=labels[i],
                        color=colors[i],
                        linestyle=line_styles[i],
                        marker=markers[i],
                        markersize=5)
        lines.append(line)

    # Add legend, title, and labels
    ax.legend(loc='best')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Enable toolbar for zooming, panning, etc.
    plt.tight_layout()

    # Add a reset button
    reset_ax = plt.axes([0.8, 0.01, 0.1, 0.04])
    reset_button = Button(reset_ax, 'Reset View')

    def reset_view(event):
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    reset_button.on_clicked(reset_view)

    # Add a save button
    save_ax = plt.axes([0.92, 0.01, 0.07, 0.04])
    save_button = Button(save_ax, 'Save')

    def save_figure(event):
        plt.savefig('line_plot.png', dpi=300, bbox_inches='tight')
        print("Figure saved as 'line_plot.png'")

    save_button.on_clicked(save_figure)

    plt.subplots_adjust(bottom=0.15)  # Make room for buttons

    # Return early if we're in dynamic update mode
    if update_function:
        plt.ion()  # Turn on interactive mode
        return fig, ax, lines

    plt.show()
    return fig, ax, lines


def speed_to_color(speed, min_speed=0.2, max_speed=3.0):
    """
    Konvertiert einen Geschwindigkeitswert in eine Farbe zwischen Rot (langsam) und Grün (schnell)

    Parameters:
    -----------
    speed : float
        Aktuelle Geschwindigkeit
    min_speed : float
        Minimale Geschwindigkeit (wird zu Rot)
    max_speed : float
        Maximale Geschwindigkeit (wird zu Grün)

    Returns:
    --------
    color : str or tuple
        Farbcode (RGB oder Hex)
    """
    # Begrenze die Geschwindigkeit auf den min-max Bereich
    speed = max(min_speed, min(max_speed, speed))

    # Normalisiere die Geschwindigkeit zu einem Wert zwischen 0 und 1
    normalized_speed = (speed - min_speed) / (max_speed - min_speed)

    # Erstelle eine Farbkarte von Rot (0) zu Grün (1)
    cmap = mcolors.LinearSegmentedColormap.from_list("speed_colors", ["red", "green"])

    # Konvertiere den normalisierten Wert in eine Farbe
    color = cmap(normalized_speed)

    return color


# Dann in deiner Hauptschleife:
if __name__ == "__main__":
    center, right, left = load_coordinates_from_file("data/Aschheim-track-data.txt")
    arrays_of_points = [left, center, right]

    # Initial boundaries
    lower_boundary = 0
    upper_boundary = 6

    # Append initial racing line and get initial speed
    racing_line, current_speed, offset = optimiere_trajektorie(center[lower_boundary:upper_boundary], 5.9)
    arrays_of_points.append(racing_line)

    # Custom labels and colors
    labels = ["Left Boundary", "Center", "Right Boundary", "Racing Line"]
    initial_colors = ["blue", "red", "green", speed_to_color(current_speed)]
    line_styles = ["-", "-", "-", "-"]
    markers = ["", "s", "", "D"]

    # Create the visualization with interactive mode
    fig, ax, lines = visualize_multiple_lines(
        arrays_of_points,
        labels=labels,
        colors=initial_colors,
        line_styles=line_styles,
        markers=markers,
        title="Corner Shape Visualization",
        x_label="X Coordinate",
        y_label="Y Coordinate",
        update_function=True  # Enable interactive updates
    )

    # Add speed text annotation
    speed_text = ax.text(0.02, 0.95, f"Speed: {current_speed:.2f} m/s",
                         transform=ax.transAxes, fontsize=12,
                         bbox=dict(facecolor='white', alpha=0.7))

    # Now we can update the racing line in a loop
    racing_line_index = 3  # The index of the racing line in arrays_of_points

    # Simulate moving through the track
    max_points = (len(center)) - 6  # Make sure we don't exceed array bounds
    previous_offset = 0

    for i in range(max_points):
        lower_boundary = i
        upper_boundary = i + 6

        # Calculate new racing line and get new speed
        new_racing_line, current_speed, previous_offset = optimiere_trajektorie(center[lower_boundary:upper_boundary], 5.9, previous_offset)

        # Update the racing line data
        if len(new_racing_line) > 0:
            new_points = np.array(new_racing_line)
            if new_points.ndim == 1:
                x = np.arange(len(new_points))
                y = new_points
            else:
                x = new_points[:, 0]
                y = new_points[:, 1]

            # Update the line data
            lines[racing_line_index].set_xdata(x)
            lines[racing_line_index].set_ydata(y)

            # Update the line color based on speed
            new_color = speed_to_color(current_speed)
            lines[racing_line_index].set_color(new_color)

            # Update the speed text
            speed_text.set_text(f"Speed: {current_speed:.2f} m/s")
            speed_text.set_bbox(dict(facecolor=new_color, alpha=0.2))

            # Adjust axes limits if needed
            ax.relim()
            ax.autoscale_view()

            # Redraw the figure
            fig.canvas.draw_idle()
            plt.pause(1)  # Pause to allow the figure to update

    # Keep the plot open after animation completes
    plt.ioff()
    plt.show()
