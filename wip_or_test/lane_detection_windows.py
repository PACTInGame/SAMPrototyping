import os

import cv2
import numpy as np
import time


class LaneDetector:
    def __init__(self):
        # Parameter für die Fahrspurerkennung
        self.num_points = 7  # Anzahl der zu erkennenden Mittelpunkte
        self.default_lane_width = 600  # Standardbreite der Fahrspur in Pixeln

    def detect_center_points(self, image):
        """
        Erkennt Mittelpunkte der Fahrspur im gegebenen Bild.

        Args:
            image: Birds-Eye-View Bild der Fahrspur

        Returns:
            list: Liste von (x, y) Mittelpunkten
        """
        if image is None or image.size == 0:
            return []

        # Bilddimensionen speichern
        h, w = image.shape[:2]

        # Kopie für Visualisierung erstellen
        vis_image = image.copy()

        # In HSV konvertieren für bessere Farbfilterung
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Bereich für weiße Farbe
        lower_white = np.array([30, 0, 70])  # Niedriger Farbton, niedrige Sättigung, hoher Helligkeitswert
        upper_white = np.array([80, 50, 255])

        # Masken für weiß und gelb erstellen
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Masken kombinieren

        # Morphologische Operationen, um Rauschen zu entfernen
        kernel = np.ones((5, 5), np.uint8)
        filtered_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)

        # Kanten mit Canny-Edge-Detection erkennen
        edges = cv2.Canny(filtered_mask, 50, 150)

        # Linien mit Hough-Transformation erkennen
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=50, minLineLength=40, maxLineGap=25)

        # Linien überprüfen und sortieren
        left_lines = []
        right_lines = []

        if lines is not None:
            mid_x = w // 2

            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Steigung berechnen (vermeidet Division durch Null)
                if x2 - x1 == 0:
                    continue

                slope = (y2 - y1) / (x2 - x1)

                # Zu steile oder zu flache Linien filtern
                if abs(slope) < 0.3 or abs(slope) > 5:
                    continue

                # Linien klassifizieren
                if slope < 0 and x1 < mid_x and x2 < mid_x:
                    left_lines.append(line[0])
                elif slope > 0 and x1 > mid_x and x2 > mid_x:
                    right_lines.append(line[0])

        # Segmentieren des Bildes in horizontale Abschnitte
        segment_height = h / (self.num_points + 1)
        y_values = [int(segment_height * (i + 1)) for i in range(self.num_points)]

        left_points = []
        right_points = []
        center_points = []

        # Für jede Höhe links- und rechtsseitige Punkte finden
        if len(left_lines) >= 2 or len(right_lines) >= 2:
            # Punkte aus Linien extrahieren
            if len(left_lines) >= 2:
                left_x = []
                left_y = []
                for line in left_lines:
                    x1, y1, x2, y2 = line
                    left_x.extend([x1, x2])
                    left_y.extend([y1, y2])
                left_poly = np.polyfit(left_y, left_x, 2)

            if len(right_lines) >= 2:
                right_x = []
                right_y = []
                for line in right_lines:
                    x1, y1, x2, y2 = line
                    right_x.extend([x1, x2])
                    right_y.extend([y1, y2])
                right_poly = np.polyfit(right_y, right_x, 2)

            # Mittelpunkte berechnen
            for y in y_values:
                if len(left_lines) >= 2 and len(right_lines) >= 2:
                    # Beide Fahrspurlinien erkannt
                    left_x_at_y = int(np.polyval(left_poly, y))
                    right_x_at_y = int(np.polyval(right_poly, y))
                    center_x = (left_x_at_y + right_x_at_y) // 2

                    left_points.append((left_x_at_y, y))
                    right_points.append((right_x_at_y, y))

                elif len(left_lines) >= 2:
                    # Nur linke Linie erkannt
                    left_x_at_y = int(np.polyval(left_poly, y))
                    center_x = left_x_at_y + self.default_lane_width // 2

                    left_points.append((left_x_at_y, y))
                    right_points.append((left_x_at_y + self.default_lane_width, y))

                elif len(right_lines) >= 2:
                    # Nur rechte Linie erkannt
                    right_x_at_y = int(np.polyval(right_poly, y))
                    center_x = right_x_at_y - self.default_lane_width // 2

                    left_points.append((right_x_at_y - self.default_lane_width, y))
                    right_points.append((right_x_at_y, y))

                center_points.append((center_x, y))

        else:
            # Keine Fahrspurlinien erkannt, Fallback-Strategie
            for y in y_values:
                center_x = w // 2
                left_x = center_x - self.default_lane_width // 2
                right_x = center_x + self.default_lane_width // 2

                left_points.append((left_x, y))
                right_points.append((right_x, y))
                center_points.append((center_x, y))

        # Ergebnisse visualisieren
        self._visualize_results(vis_image, left_points, right_points, center_points)
        cv2.imshow("Lane Detection", vis_image)
        cv2.imshow("Binary Mask", filtered_mask)
        cv2.imshow("Edges", edges)

        return center_points, vis_image

    def _visualize_results(self, image, left_points, right_points, center_points):
        """Visualisiert die erkannten Fahrspurgrenzen und Mittelpunkte."""
        # Fahrspurlinien zeichnen
        if len(left_points) >= 2:
            # Punkte in NumPy-Array umwandeln für Polyline
            left_points_array = np.array(left_points, dtype=np.int32)
            cv2.polylines(image, [left_points_array], False, (255, 0, 0), 2)

        if len(right_points) >= 2:
            right_points_array = np.array(right_points, dtype=np.int32)
            cv2.polylines(image, [right_points_array], False, (0, 0, 255), 2)

        # Mittelpunkte als grüne Kreise zeichnen
        for point in center_points:
            cv2.circle(image, point, 5, (0, 255, 0), -1)

        # Mittellinie zeichnen
        if len(center_points) >= 2:
            center_points_array = np.array(center_points, dtype=np.int32)
            cv2.polylines(image, [center_points_array], False, (0, 255, 0), 2)


def main():
    detector = LaneDetector()

    downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
    image_path = os.path.join(downloads_folder, "vdi_24_after_finals-frame94.jpg")  # Ändere "dein_bild.jpg" zum tatsächlichen Dateinamen

    # Bild einlesen
    frame = cv2.imread(image_path)

    # FPS-Zähler
    fps_counter = 0
    fps = 0
    start_time = time.time()

    print("Drücke 'ESC' zum Beenden, 's' zum Speichern des aktuellen Frames.")

    while True:
        cv2.imshow('Bild aus Downloads', frame)


        # Region of Interest definieren (untere Hälfte des Bildes)
        h, w = frame.shape[:2]
        roi = frame[int(h / 2):h, 0:w]


        # Mittelpunkte erkennen
        center_points, result_image = detector.detect_center_points(roi)

        # FPS berechnen und anzeigen
        fps_counter += 1
        current_time = time.time()
        if current_time - start_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            start_time = current_time

        # FPS im Bild anzeigen
        cv2.putText(result_image, f'FPS: {fps}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Original Bild mit ROI-Markierung anzeigen
        cv2.rectangle(frame, (0, int(h / 2)), (w, h), (0, 255, 0), 2)
        cv2.imshow('Original mit ROI', frame)

        # Auf Tastendruck warten
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC-Taste
            break
        elif key == ord('s'):  # 's'-Taste zum Speichern
            timestamp = int(time.time())
            cv2.imwrite(f'lane_detection_{timestamp}.png', result_image)
            print(f'Bild gespeichert als lane_detection_{timestamp}.png')

    # Aufräumen
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()