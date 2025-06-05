import cv2
import numpy as np
from typing import Tuple, Optional
import time
from dataclasses import dataclass


@dataclass
class PerspectiveParams:
    """Parameter für Vogelperspektive-Transformation"""
    # Quellpunkte (Trapez auf der Straße) - als Verhältnisse zur Bildgröße
    src_top_left_ratio: Tuple[float, float] = (0.3, 0.0)  # Links oben
    src_top_right_ratio: Tuple[float, float] = (0.7, 0.0)  # Rechts oben
    src_bottom_left_ratio: Tuple[float, float] = (0, 0.6)  # Links unten
    src_bottom_right_ratio: Tuple[float, float] = (1, 0.6)  # Rechts unten

    # Zielpunkte (Rechteck in Vogelperspektive) - als Verhältnisse
    dst_top_left_ratio: Tuple[float, float] = (0.25, 0.0)  # Links oben
    dst_top_right_ratio: Tuple[float, float] = (0.75, 0.0)  # Rechts oben
    dst_bottom_left_ratio: Tuple[float, float] = (0.25, 1.0)  # Links unten
    dst_bottom_right_ratio: Tuple[float, float] = (0.75, 1.0)  # Rechts unten


@dataclass
class ProcessingParams:
    """Optimierte Parameter für Spurerkennung"""
    # ROI Parameter
    roi_top_ratio: float = 0.5
    roi_bottom_ratio: float = 0.9

    # Vorverarbeitung
    gaussian_kernel: int = 3
    clahe_clip_limit: float = 2.0
    clahe_tile_size: Tuple[int, int] = (4, 4)

    # White mask - HAUPTMETHODE
    white_threshold_low: int = 190  # Etwas entspannter
    white_threshold_high: int = 255
    white_mask_weight: float = 0.8  # Hohe Priorität für white mask

    # Adaptive Thresholding - nur als Ergänzung
    adaptive_max_value: int = 255
    adaptive_block_size: int = 15
    adaptive_c: int = -3
    adaptive_weight: float = 0.2  # Niedrige Priorität

    # Morphologie - sehr minimal
    morph_kernel_size: int = 2
    morph_iterations: int = 1

    # Rauschfilterung - weniger aggressiv
    min_contour_area: int = 30  # Reduziert für Kurven
    max_contour_area: int = 3000  # Erhöht
    max_width: int = 80  # Erhöht für Kurven
    # Aspect ratio entfernt - zu restriktiv für Kurven


class PerspectiveTransformer:
    """Klasse für Vogelperspektive-Transformation"""

    def __init__(self, params: PerspectiveParams = None):
        self.params = params or PerspectiveParams()
        self.M = None  # Transformationsmatrix
        self.M_inv = None  # Inverse Transformationsmatrix
        self.image_shape = None

    def calculate_transform_matrices(self, image_shape: Tuple[int, int]):
        """Berechnet Transformationsmatrizen basierend auf Bildgröße"""
        height, width = image_shape[:2]

        # Quellpunkte (Trapez auf Straße)
        src_points = np.float32([
            [width * self.params.src_top_left_ratio[0], height * self.params.src_top_left_ratio[1]],
            [width * self.params.src_top_right_ratio[0], height * self.params.src_top_right_ratio[1]],
            [width * self.params.src_bottom_right_ratio[0], height * self.params.src_bottom_right_ratio[1]],
            [width * self.params.src_bottom_left_ratio[0], height * self.params.src_bottom_left_ratio[1]]
        ])

        # Zielpunkte (Rechteck in Vogelperspektive)
        dst_points = np.float32([
            [width * self.params.dst_top_left_ratio[0], height * self.params.dst_top_left_ratio[1]],
            [width * self.params.dst_top_right_ratio[0], height * self.params.dst_top_right_ratio[1]],
            [width * self.params.dst_bottom_right_ratio[0], height * self.params.dst_bottom_right_ratio[1]],
            [width * self.params.dst_bottom_left_ratio[0], height * self.params.dst_bottom_left_ratio[1]]
        ])

        # Transformationsmatrizen berechnen
        self.M = cv2.getPerspectiveTransform(src_points, dst_points)
        self.M_inv = cv2.getPerspectiveTransform(dst_points, src_points)
        self.image_shape = image_shape

        return src_points, dst_points

    def transform_to_birds_eye(self, image: np.ndarray) -> np.ndarray:
        """Transformiert Bild in Vogelperspektive"""
        if self.M is None or self.image_shape != image.shape:
            self.calculate_transform_matrices(image.shape)

        return cv2.warpPerspective(image, self.M, (image.shape[1], image.shape[0]))

    def transform_to_original(self, birds_eye_image: np.ndarray) -> np.ndarray:
        """Transformiert zurück zur ursprünglichen Perspektive"""
        if self.M_inv is None:
            raise ValueError("Transformationsmatrizen nicht initialisiert")

        return cv2.warpPerspective(birds_eye_image, self.M_inv,
                                   (birds_eye_image.shape[1], birds_eye_image.shape[0]))

    def get_transform_visualization(self, image: np.ndarray) -> np.ndarray:
        """Erstellt Visualisierung der Transformationspunkte"""
        if self.image_shape != image.shape:
            src_points, _ = self.calculate_transform_matrices(image.shape)
        else:
            height, width = image.shape[:2]
            src_points = np.float32([
                [width * self.params.src_top_left_ratio[0], height * self.params.src_top_left_ratio[1]],
                [width * self.params.src_top_right_ratio[0], height * self.params.src_top_right_ratio[1]],
                [width * self.params.src_bottom_right_ratio[0], height * self.params.src_bottom_right_ratio[1]],
                [width * self.params.src_bottom_left_ratio[0], height * self.params.src_bottom_left_ratio[1]]
            ])

        vis_image = image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)

        # Zeichne Transformationsbereich
        pts = src_points.astype(np.int32)
        cv2.polylines(vis_image, [pts], True, (0, 255, 0), 2)

        # Markiere Eckpunkte
        for i, point in enumerate(pts):
            cv2.circle(vis_image, tuple(point), 5, (0, 0, 255), -1)
            cv2.putText(vis_image, str(i + 1), tuple(point + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return vis_image


class OptimizedLaneDetector:
    """Optimierte Spurerkennung mit Vogelperspektive und white-mask Priorität"""

    def __init__(self, processing_params: ProcessingParams = None,
                 perspective_params: PerspectiveParams = None):
        self.processing_params = processing_params or ProcessingParams()
        self.perspective_transformer = PerspectiveTransformer(perspective_params)
        self.debug_mode = True
        self.step_images = {}

        # CLAHE Object
        self.clahe = cv2.createCLAHE(
            clipLimit=self.processing_params.clahe_clip_limit,
            tileGridSize=self.processing_params.clahe_tile_size
        )

        # Morphologische Kernel
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.processing_params.morph_kernel_size, self.processing_params.morph_kernel_size)
        )

    def define_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """ROI definieren"""
        height, width = frame.shape[:2]

        roi_top = int(height * self.processing_params.roi_top_ratio)
        roi_bottom = int(height * self.processing_params.roi_bottom_ratio)

        if self.debug_mode:
            roi_vis = frame.copy()
            if len(roi_vis.shape) == 2:
                roi_vis = cv2.cvtColor(roi_vis, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(roi_vis, (0, roi_top), (width, roi_bottom), (0, 255, 0), 2)
            self.step_images['01_roi'] = roi_vis

        roi = frame[roi_top:roi_bottom, :]
        roi_coords = (0, roi_top, width, roi_bottom)

        return roi, roi_coords

    def preprocessing(self, frame: np.ndarray) -> np.ndarray:
        """Schonende Vorverarbeitung"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        if self.debug_mode:
            self.step_images['02_gray'] = gray

        # Minimale Glättung
        blurred = cv2.GaussianBlur(gray,
                                   (self.processing_params.gaussian_kernel,
                                    self.processing_params.gaussian_kernel), 0)

        if self.debug_mode:
            self.step_images['02b_blurred'] = blurred

        return blurred

    def white_mask_extraction(self, frame: np.ndarray) -> np.ndarray:
        """Hauptmethode: Extraktion weißer Spurmarkierungen"""
        _, white_mask = cv2.threshold(frame,
                                      self.processing_params.white_threshold_low,
                                      self.processing_params.white_threshold_high,
                                      cv2.THRESH_BINARY)

        if self.debug_mode:
            self.step_images['03_white_mask_primary'] = white_mask

        return white_mask

    def supplementary_detection(self, frame: np.ndarray) -> np.ndarray:
        """Ergänzende Detektion (niedrige Priorität)"""
        adaptive_thresh = cv2.adaptiveThreshold(
            frame,
            self.processing_params.adaptive_max_value,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.processing_params.adaptive_block_size,
            self.processing_params.adaptive_c
        )

        if self.debug_mode:
            self.step_images['04_adaptive_supplement'] = adaptive_thresh

        return adaptive_thresh

    def weighted_combination(self, white_mask: np.ndarray,
                             supplementary: np.ndarray) -> np.ndarray:
        """Gewichtete Kombination mit Priorität für white_mask"""
        # White mask hat höchste Priorität
        white_weighted = (white_mask * self.processing_params.white_mask_weight).astype(np.uint8)

        # Supplementary nur dort wo white_mask schwach ist
        supplement_weighted = (supplementary * self.processing_params.adaptive_weight).astype(np.uint8)

        # Kombiniere, aber white_mask dominiert
        combined = cv2.addWeighted(white_weighted, 1.0, supplement_weighted, 1.0, 0)

        # Schwellwert anwenden
        _, combined_binary = cv2.threshold(combined, 100, 255, cv2.THRESH_BINARY)

        if self.debug_mode:
            self.step_images['05_weighted_combination'] = combined_binary

        return combined_binary

    def gentle_noise_filtering(self, binary_image: np.ndarray) -> np.ndarray:
        """Sanfte Rauschfilterung ohne aggressive Formkriterien"""
        # Konturen finden
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if self.debug_mode:
            contour_debug = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_debug, contours, -1, (0, 255, 0), 1)
            self.step_images['06_all_contours'] = contour_debug

        # Sanfte Filterung
        filtered_mask = np.zeros_like(binary_image)
        valid_contours = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Nur grundlegende Größenfilterung
            if (self.processing_params.min_contour_area <= area <=
                    self.processing_params.max_contour_area):

                # Sehr sanfte Breitenfilterung
                x, y, w, h = cv2.boundingRect(contour)
                if w <= self.processing_params.max_width:
                    cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
                    valid_contours.append(contour)

        if self.debug_mode:
            valid_debug = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(valid_debug, valid_contours, -1, (0, 0, 255), 2)
            self.step_images['07_filtered_contours'] = valid_debug

        return filtered_mask

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Hauptverarbeitung mit Vogelperspektive"""
        self.step_images.clear()
        original = frame.copy()

        # 1. ROI definieren
        roi_frame, roi_coords = self.define_roi(frame)

        # 2. Perspektive-Transformation zeigen
        if self.debug_mode:
            perspective_vis = self.perspective_transformer.get_transform_visualization(roi_frame)
            self.step_images['02c_perspective_points'] = perspective_vis

        # 3. Vorverarbeitung
        preprocessed = self.preprocessing(roi_frame)

        # 4. Vogelperspektive anwenden
        birds_eye = self.perspective_transformer.transform_to_birds_eye(preprocessed)

        if self.debug_mode:
            self.step_images['03_birds_eye'] = birds_eye

        # 5. White mask (Hauptmethode)
        white_mask = self.white_mask_extraction(birds_eye)

        # 6. Ergänzende Detektion
        supplementary = self.supplementary_detection(birds_eye)

        # 7. Gewichtete Kombination
        combined = self.weighted_combination(white_mask, supplementary)

        # 8. Sanfte Rauschfilterung
        filtered = self.gentle_noise_filtering(combined)

        # 9. Minimale Morphologie
        if self.processing_params.morph_iterations > 0:
            final_birds_eye = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE,
                                               self.morph_kernel,
                                               iterations=self.processing_params.morph_iterations)
        else:
            final_birds_eye = filtered

        if self.debug_mode:
            self.step_images['08_final_birds_eye'] = final_birds_eye

        # 10. Zurück zur ursprünglichen Perspektive
        final_roi = self.perspective_transformer.transform_to_original(final_birds_eye)

        if self.debug_mode:
            self.step_images['09_back_to_original'] = final_roi

        # 11. Vollbild erstellen
        full_output = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        full_output[roi_coords[1]:roi_coords[3], roi_coords[0]:roi_coords[2]] = final_roi

        return full_output, final_birds_eye, self.step_images


class AdvancedLaneSystem:
    """Erweiterte Spurerkennung mit allen Features"""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.detector = OptimizedLaneDetector()
        self.running = False
        self.show_debug = True
        self.show_birds_eye = False

    def initialize_camera(self) -> bool:
        """Kamera initialisieren"""
        #self.cap = cv2.VideoCapture(self.camera_index)
        self.cap = cv2.VideoCapture("camera_car.mp4")
        if not self.cap.isOpened():
            print(f"Fehler: Kamera {self.camera_index} kann nicht geöffnet werden")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        return True

    def run(self):
        """Hauptschleife"""
        if not self.initialize_camera():
            return

        print("=== ERWEITERTE SPURERKENNUNG ===")
        print("Steuerung:")
        print("  q: Beenden")
        print("  d: Debug-Fenster ein/aus")
        print("  b: Vogelperspektive ein/aus")
        print("  s: Screenshot speichern")
        print("  p: Parameter anzeigen")
        print("  r: Parameter zur Laufzeit ändern")
        print("  1-9: Einzelne Debug-Schritte")

        self.running = True

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Verarbeitung
            result_mask, birds_eye_result, debug_images = self.detector.process_frame(frame)

            # Hauptfenster
            cv2.imshow('Original', frame)
            cv2.imshow('Erkannte Spurmarkierungen', result_mask)

            # Vogelperspektive
            if self.show_birds_eye:
                cv2.imshow('Vogelperspektive', birds_eye_result)

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
            elif key == ord('b'):
                self.show_birds_eye = not self.show_birds_eye
                if not self.show_birds_eye:
                    cv2.destroyWindow('Vogelperspektive')
            elif key == ord('s'):
                self.save_all_images(frame, result_mask, birds_eye_result, debug_images)
            elif key == ord('p'):
                self.print_parameters()
            elif key == ord('r'):
                self.runtime_parameter_adjustment()
            elif ord('1') <= key <= ord('9'):
                self.show_single_debug_step(debug_images, key - ord('1'))

        self.cleanup()

    def runtime_parameter_adjustment(self):
        """Laufzeit-Parameteranpassung"""
        print("\n=== PARAMETER ANPASSUNG ===")
        print("Aktuelle Werte:")
        params = self.detector.processing_params
        print(f"1. White Threshold Low: {params.white_threshold_low}")
        print(f"2. Adaptive C: {params.adaptive_c}")
        print(f"3. ROI Top Ratio: {params.roi_top_ratio}")
        print(f"4. White Mask Weight: {params.white_mask_weight}")

        try:
            choice = input("Parameter ändern (1-4, Enter für zurück): ")
            if choice == '1':
                new_val = int(input(f"Neuer White Threshold Low ({params.white_threshold_low}): "))
                params.white_threshold_low = max(0, min(255, new_val))
            elif choice == '2':
                new_val = int(input(f"Neuer Adaptive C ({params.adaptive_c}): "))
                params.adaptive_c = max(-20, min(20, new_val))
            elif choice == '3':
                new_val = float(input(f"Neue ROI Top Ratio ({params.roi_top_ratio}): "))
                params.roi_top_ratio = max(0.0, min(1.0, new_val))
            elif choice == '4':
                new_val = float(input(f"Neue White Mask Weight ({params.white_mask_weight}): "))
                params.white_mask_weight = max(0.0, min(1.0, new_val))
        except:
            print("Ungültige Eingabe")

    def show_debug_windows(self, debug_images: dict):
        """Debug-Fenster anzeigen"""
        for name, image in debug_images.items():
            cv2.imshow(f'Debug: {name}', image)

    def close_debug_windows(self, debug_images: dict):
        """Debug-Fenster schließen"""
        for name in debug_images.keys():
            cv2.destroyWindow(f'Debug: {name}')

    def show_single_debug_step(self, debug_images: dict, step: int):
        """Einzelnen Debug-Schritt anzeigen"""
        debug_list = list(debug_images.items())
        if 0 <= step < len(debug_list):
            name, image = debug_list[step]
            cv2.imshow(f'Step {step + 1}: {name}', image)

    def print_parameters(self):
        """Parameter ausgeben"""
        print("\n=== AKTUELLE PARAMETER ===")
        params = self.detector.processing_params
        print(f"White Threshold Low: {params.white_threshold_low}")
        print(f"White Mask Weight: {params.white_mask_weight}")
        print(f"Adaptive Weight: {params.adaptive_weight}")
        print(f"ROI Top/Bottom: {params.roi_top_ratio}/{params.roi_bottom_ratio}")
        print(f"Min/Max Contour Area: {params.min_contour_area}/{params.max_contour_area}")

    def save_all_images(self, original, result, birds_eye, debug_images):
        """Alle Bilder speichern"""
        timestamp = int(time.time())
        cv2.imwrite(f'lane_original_{timestamp}.jpg', original)
        cv2.imwrite(f'lane_result_{timestamp}.jpg', result)
        cv2.imwrite(f'lane_birds_eye_{timestamp}.jpg', birds_eye)

        for name, image in debug_images.items():
            cv2.imwrite(f'debug_{name}_{timestamp}.jpg', image)
        print(f"Alle Bilder gespeichert: {timestamp}")

    def cleanup(self):
        """Aufräumen"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    system = AdvancedLaneSystem(camera_index=2)
    system.run()


if __name__ == "__main__":
    main()