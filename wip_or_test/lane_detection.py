import cv2
import numpy as np
from typing import Tuple, Optional
import time
from dataclasses import dataclass


@dataclass
class ProcessingParams:
    """Optimierte Parameter für Spurerkennung"""
    # ROI Parameter - KRITISCH!
    roi_top_ratio: float = 0.5  # Erhöht von 0.4 - fokussiert auf nähere Spuren
    roi_bottom_ratio: float = 0.9  # Reduziert von 0.95 - vermeidet Motorhaube

    # Vorverarbeitung
    gaussian_kernel: int = 3  # Reduziert von 5 - weniger Unschärfe
    clahe_clip_limit: float = 2.0  # Reduziert von 3.0 - weniger aggressive Kontrastverstärkung
    clahe_tile_size: Tuple[int, int] = (4, 4)  # Kleinere Tiles für lokale Anpassung

    # Für weiße Spurmarkierungen optimiert
    white_threshold_low: int = 200  # NEU: Direkter Schwellwert für weiße Markierungen
    white_threshold_high: int = 255

    # Adaptive Thresholding - angepasst
    adaptive_max_value: int = 255
    adaptive_block_size: int = 15  # Erhöht von 11 - größere lokale Bereiche
    adaptive_c: int = -2  # Negativ! Für weiße Objekte auf dunklem Hintergrund

    # Canny - konservativer
    canny_low: int = 50
    canny_high: int = 100  # Reduziert von 150

    # Morphologie - weniger aggressiv
    morph_kernel_size: int = 2  # Reduziert von 3
    morph_iterations: int = 1  # Reduziert von 2

    # Spurmarkierungsfilter - spezifisch für Linien
    min_contour_area: int = 100  # Erhöht von 50
    max_contour_area: int = 2000  # Reduziert von 5000
    min_aspect_ratio: float = 3.0  # NEU: Mindest-Seitenverhältnis für Linien
    max_width: int = 50  # NEU: Maximale Breite für Spurmarkierungen


class DebugLaneDetector:
    """Debugging-Version der Spurerkennung"""

    def __init__(self, params: ProcessingParams = None):
        self.params = params or ProcessingParams()
        self.debug_mode = True
        self.step_images = {}

        # CLAHE Object
        self.clahe = cv2.createCLAHE(
            clipLimit=self.params.clahe_clip_limit,
            tileGridSize=self.params.clahe_tile_size
        )

        # Morphologische Kernel
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.params.morph_kernel_size, self.params.morph_kernel_size)
        )

    def define_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """ROI mit Visualisierung"""
        height, width = frame.shape[:2]

        roi_top = int(height * self.params.roi_top_ratio)
        roi_bottom = int(height * self.params.roi_bottom_ratio)

        # ROI visualisieren
        if self.debug_mode:
            roi_vis = frame.copy()
            cv2.rectangle(roi_vis, (0, roi_top), (width, roi_bottom), (0, 255, 0), 2)
            cv2.putText(roi_vis, 'ROI', (10, roi_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self.step_images['01_roi'] = roi_vis

        roi = frame[roi_top:roi_bottom, :]
        roi_coords = (0, roi_top, width, roi_bottom)

        return roi, roi_coords

    def white_line_enhancement(self, frame: np.ndarray) -> np.ndarray:
        """Spezielle Verbesserung für weiße Spurmarkierungen"""
        # Direkte Schwellwertbildung für weiße Bereiche
        _, white_mask = cv2.threshold(frame,
                                      self.params.white_threshold_low,
                                      self.params.white_threshold_high,
                                      cv2.THRESH_BINARY)

        if self.debug_mode:
            self.step_images['03_white_mask'] = white_mask

        return white_mask

    def improved_preprocessing(self, frame: np.ndarray) -> np.ndarray:
        """Verbesserte Vorverarbeitung"""
        # Graustufen
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        if self.debug_mode:
            self.step_images['02_gray'] = gray

        # Leichte Glättung
        blurred = cv2.GaussianBlur(gray,
                                   (self.params.gaussian_kernel, self.params.gaussian_kernel),
                                   0)

        # CLAHE nur bei Bedarf (wenn Bild zu dunkel/hell)
        mean_brightness = np.mean(blurred)
        if mean_brightness < 100 or mean_brightness > 200:
            enhanced = self.clahe.apply(blurred)
        else:
            enhanced = blurred

        if self.debug_mode:
            self.step_images['02b_enhanced'] = enhanced

        return enhanced

    def advanced_thresholding(self, frame: np.ndarray) -> np.ndarray:
        """Mehrstufige Schwellwertbildung speziell für Spurmarkierungen"""

        # 1. Weiße Linien direkt extrahieren
        white_mask = self.white_line_enhancement(frame)

        # 2. Adaptive Thresholding für lokale Variationen
        adaptive_thresh = cv2.adaptiveThreshold(
            frame,
            self.params.adaptive_max_value,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.params.adaptive_block_size,
            self.params.adaptive_c
        )

        if self.debug_mode:
            self.step_images['04_adaptive'] = adaptive_thresh

        # 3. Kombiniere beide Methoden
        combined = cv2.bitwise_or(white_mask, adaptive_thresh)

        if self.debug_mode:
            self.step_images['05_combined_thresh'] = combined

        return combined

    def lane_specific_filtering(self, binary_image: np.ndarray) -> np.ndarray:
        """Filterung speziell für Spurmarkierungen"""
        # Konturen finden
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Debug-Bild für Konturen
        if self.debug_mode:
            contour_debug = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_debug, contours, -1, (0, 255, 0), 2)
            self.step_images['06_all_contours'] = contour_debug

        # Gefilterte Maske erstellen
        filtered_mask = np.zeros_like(binary_image)
        valid_contours = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Grundlegende Größenfilterung
            if not (self.params.min_contour_area <= area <= self.params.max_contour_area):
                continue

            # Bounding Rectangle für weitere Analyse
            x, y, w, h = cv2.boundingRect(contour)

            # Spurmarkierungen sind typischerweise länger als breit
            aspect_ratio = h / max(w, 1)  # Höhe/Breite

            # Filter für spurmarkierungsähnliche Formen
            if (aspect_ratio >= self.params.min_aspect_ratio and
                    w <= self.params.max_width):
                cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
                valid_contours.append(contour)

        # Debug-Ausgabe
        if self.debug_mode:
            valid_contour_debug = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(valid_contour_debug, valid_contours, -1, (0, 0, 255), 2)
            self.step_images['07_filtered_contours'] = valid_contour_debug
            self.step_images['08_final_mask'] = filtered_mask

        return filtered_mask

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Hauptverarbeitung mit Debug-Ausgaben"""
        self.step_images.clear()
        original = frame.copy()

        # 1. ROI definieren
        roi_frame, roi_coords = self.define_roi(frame)

        # 2. Vorverarbeitung
        preprocessed = self.improved_preprocessing(roi_frame)

        # 3. Spezielle Spurmarkierungs-Schwellwertbildung
        thresholded = self.advanced_thresholding(preprocessed)

        # 4. Spurmarkierungs-spezifische Filterung
        final_mask = self.lane_specific_filtering(thresholded)

        # 5. Minimale morphologische Glättung
        if self.params.morph_iterations > 0:
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE,
                                          self.morph_kernel,
                                          iterations=self.params.morph_iterations)

        # Vollbild-Ausgabe erstellen
        full_output = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        full_output[roi_coords[1]:roi_coords[3], roi_coords[0]:roi_coords[2]] = final_mask

        return full_output, self.step_images


class DebugLaneSystem:
    """Debug-System für Spurerkennung"""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.detector = DebugLaneDetector()
        self.running = False
        self.show_debug = True

    def initialize_camera(self) -> bool:
        """Kamera initialisieren"""
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            print(f"Fehler: Kamera {self.camera_index} kann nicht geöffnet werden")
            return False

        # Kameraeinstellungen
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        return True

    def run(self):
        """Hauptschleife mit Debug-Anzeige"""
        if not self.initialize_camera():
            return

        print("=== DEBUG SPURERKENNUNG ===")
        print("Steuerung:")
        print("  q: Beenden")
        print("  d: Debug-Fenster ein/aus")
        print("  s: Screenshot speichern")
        print("  p: Parameter anzeigen")
        print("  1-8: Einzelne Debug-Schritte anzeigen")

        self.running = True

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Verarbeitung
            result_mask, debug_images = self.detector.process_frame(frame)

            # Hauptfenster immer anzeigen
            cv2.imshow('Original', frame)
            cv2.imshow('Erkannte Spurmarkierungen', result_mask)

            # Debug-Fenster
            if self.show_debug and debug_images:
                self.show_debug_windows(debug_images)

            # Tastatureingabe
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.show_debug = not self.show_debug
                if not self.show_debug:
                    self.close_debug_windows(debug_images)
            elif key == ord('s'):
                self.save_debug_images(frame, result_mask, debug_images)
            elif key == ord('p'):
                self.print_parameters()
            elif ord('1') <= key <= ord('8'):
                self.show_single_debug_step(debug_images, key - ord('1'))

        self.cleanup()

    def show_debug_windows(self, debug_images: dict):
        """Zeigt alle Debug-Fenster"""
        for name, image in debug_images.items():
            cv2.imshow(f'Debug: {name}', image)

    def close_debug_windows(self, debug_images: dict):
        """Schließt Debug-Fenster"""
        for name in debug_images.keys():
            cv2.destroyWindow(f'Debug: {name}')

    def show_single_debug_step(self, debug_images: dict, step: int):
        """Zeigt einen einzelnen Debug-Schritt"""
        debug_list = list(debug_images.items())
        if 0 <= step < len(debug_list):
            name, image = debug_list[step]
            cv2.imshow(f'Debug Step {step + 1}: {name}', image)

    def print_parameters(self):
        """Gibt aktuelle Parameter aus"""
        print("\n=== AKTUELLE PARAMETER ===")
        params = self.detector.params
        print(f"ROI Top Ratio: {params.roi_top_ratio}")
        print(f"ROI Bottom Ratio: {params.roi_bottom_ratio}")
        print(f"White Threshold Low: {params.white_threshold_low}")
        print(f"Adaptive Block Size: {params.adaptive_block_size}")
        print(f"Adaptive C: {params.adaptive_c}")
        print(f"Min Contour Area: {params.min_contour_area}")
        print(f"Max Contour Area: {params.max_contour_area}")
        print(f"Min Aspect Ratio: {params.min_aspect_ratio}")
        print(f"Max Width: {params.max_width}")

    def save_debug_images(self, original, result, debug_images):
        """Speichert alle Debug-Bilder"""
        timestamp = int(time.time())
        cv2.imwrite(f'debug_original_{timestamp}.jpg', original)
        cv2.imwrite(f'debug_result_{timestamp}.jpg', result)

        for name, image in debug_images.items():
            cv2.imwrite(f'debug_{name}_{timestamp}.jpg', image)
        print(f"Debug-Bilder gespeichert: {timestamp}")

    def cleanup(self):
        """Aufräumen"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    system = DebugLaneSystem(camera_index=0)
    system.run()


if __name__ == "__main__":
    main()