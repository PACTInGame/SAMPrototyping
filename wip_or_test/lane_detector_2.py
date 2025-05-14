import cv2
import numpy as np
import time
from collections import deque
import os


class LaneDetector:
    def __init__(self, lane_width, roi_vertices, memory_size=10):
        """
        Initialize the lane detector.

        Parameters:
        lane_width: Width of lane in pixels
        roi_vertices: Region of interest vertices
        memory_size: Number of frames to keep in memory for smoothing
        """
        self.lane_width = lane_width
        self.roi_vertices = roi_vertices
        self.left_lane_memory = deque(maxlen=memory_size)
        self.right_lane_memory = deque(maxlen=memory_size)
        self.center_memory = deque(maxlen=memory_size)

        # Default threshold values
        self.white_threshold = 200
        self.load_thresholds()

        # Set up perspective transform for bird's-eye view
        self.update_perspective_transform()

    def load_thresholds(self):
        """Load threshold values from file if exists"""
        if os.path.exists("lane_thresholds.txt"):
            try:
                with open("lane_thresholds.txt", "r") as f:
                    self.white_threshold = int(f.readline().strip())
                print(f"Loaded thresholds: white={self.white_threshold}")
            except:
                print("Error loading thresholds, using defaults")

    def save_thresholds(self):
        """Save threshold values to file"""
        with open("lane_thresholds.txt", "w") as f:
            f.write(f"{self.white_threshold}\n")

    def update_perspective_transform(self):
        """Update the perspective transformation matrices based on ROI."""
        self.src_points = np.float32([
            [self.roi_vertices[0][0][0], self.roi_vertices[0][0][1]],  # Bottom-left
            [self.roi_vertices[0][1][0], self.roi_vertices[0][1][1]],  # Top-left
            [self.roi_vertices[0][2][0], self.roi_vertices[0][2][1]],  # Top-right
            [self.roi_vertices[0][3][0], self.roi_vertices[0][3][1]]  # Bottom-right
        ])

        self.dst_points = np.float32([
            [100, 500],  # Bottom-left
            [100, 0],  # Top-left
            [700, 0],  # Top-right
            [700, 500]  # Bottom-right
        ])

        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.Minv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

    def update_roi(self, roi_vertices):
        """Update the ROI and perspective transform."""
        self.roi_vertices = roi_vertices
        self.update_perspective_transform()

    def detect_lanes_and_center(self, frame):
        """
        Detect lane lines and calculate center points.

        Returns:
        center_points: Array of center points
        visualization_img: Visualization of the detection
        """
        # Create a copy for visualization
        visualization_img = frame.copy()

        # Apply perspective transform to get bird's-eye view
        warped = cv2.warpPerspective(frame, self.M, (800, 600))
        warped_vis = warped.copy()

        # Extract white lines using color thresholding
        # Convert to HLS color space
        hls = cv2.cvtColor(warped, cv2.COLOR_BGR2HLS)

        # Threshold for white lanes (Lightness channel)
        l_channel = hls[:, :, 1]
        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel > self.white_threshold)] = 255

        # White line detection only (removed yellow detection as requested)
        combined_binary = l_binary.copy()

        # Apply Gaussian blur and edge detection
        blur = cv2.GaussianBlur(combined_binary, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # Apply Hough Transform to detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=100)

        # Handle case where no lines are detected
        if lines is None:
            cv2.putText(visualization_img, "No lanes detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return [], visualization_img

        # Separate left and right lane points
        left_lane_points = []
        right_lane_points = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate slope
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)

                # Filter out horizontal lines
                if abs(slope) < 0.1:
                    continue

                # Classify as left or right lane based on position
                if x1 < warped.shape[1] / 2:  # Left half
                    left_lane_points.append((x1, y1))
                    left_lane_points.append((x2, y2))
                else:  # Right half
                    right_lane_points.append((x1, y1))
                    right_lane_points.append((x2, y2))

        # Fit polynomials to lane points
        left_fit = None
        right_fit = None

        if left_lane_points:
            left_x = [point[0] for point in left_lane_points]
            left_y = [point[1] for point in left_lane_points]
            if len(left_x) >= 2:
                left_fit = np.polyfit(left_y, left_x, 2)
                self.left_lane_memory.append(left_fit)

        if right_lane_points:
            right_x = [point[0] for point in right_lane_points]
            right_y = [point[1] for point in right_lane_points]
            if len(right_x) >= 2:
                right_fit = np.polyfit(right_y, right_x, 2)
                self.right_lane_memory.append(right_fit)

        # Use memory if current detection isn't good
        if (left_fit is None or len(left_lane_points) < 4) and self.left_lane_memory:
            left_fit = np.mean(self.left_lane_memory, axis=0)

        if (right_fit is None or len(right_lane_points) < 4) and self.right_lane_memory:
            right_fit = np.mean(self.right_lane_memory, axis=0)

        # Generate points for visualization
        ploty = np.linspace(0, warped.shape[0] - 1, 50)
        left_fitx = []
        right_fitx = []

        if left_fit is not None:
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            # Ensure points are within image bounds
            left_fitx = np.clip(left_fitx, 0, warped.shape[1] - 1)

        if right_fit is not None:
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
            right_fitx = np.clip(right_fitx, 0, warped.shape[1] - 1)

        # Handle missing lanes
        if len(left_fitx) == 0 and len(right_fitx) > 0:
            left_fitx = right_fitx - self.lane_width
            left_fitx = np.clip(left_fitx, 0, warped.shape[1] - 1)
        elif len(right_fitx) == 0 and len(left_fitx) > 0:
            right_fitx = left_fitx + self.lane_width
            right_fitx = np.clip(right_fitx, 0, warped.shape[1] - 1)

        # Calculate center line
        center_fitx = []
        center_points_warped = []

        if len(left_fitx) > 0 and len(right_fitx) > 0:
            center_fitx = [(left + right) / 2 for left, right in zip(left_fitx, right_fitx)]
            center_points_warped = [(int(x), int(y)) for x, y in zip(center_fitx, ploty)]

        # Transform center points back to original image
        center_points = []
        for point in center_points_warped:
            # Fix: Correct format for perspectiveTransform
            pts = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
            warped_pt = cv2.perspectiveTransform(pts, self.Minv)
            center_points.append((int(warped_pt[0][0][0]), int(warped_pt[0][0][1])))

        # Visualize on warped image
        if len(left_fitx) > 0:
            points = np.array([np.transpose(np.vstack([left_fitx, ploty]))], dtype=np.int32)
            cv2.polylines(warped_vis, points, False, (0, 0, 255), 2)

        if len(right_fitx) > 0:
            points = np.array([np.transpose(np.vstack([right_fitx, ploty]))], dtype=np.int32)
            cv2.polylines(warped_vis, points, False, (0, 0, 255), 2)

        if len(center_fitx) > 0:
            points = np.array([np.transpose(np.vstack([center_fitx, ploty]))], dtype=np.int32)
            cv2.polylines(warped_vis, points, False, (0, 255, 0), 2)

        # Show warped view for debugging
        cv2.imshow('Bird\'s-Eye View', warped_vis)
        cv2.imshow('Lane Thresholding', combined_binary)

        # Create final visualization
        result_overlay = cv2.warpPerspective(warped_vis, self.Minv, (frame.shape[1], frame.shape[0]))
        result = cv2.addWeighted(visualization_img, 1, result_overlay, 0.6, 0)

        # Draw ROI
        cv2.polylines(result, self.roi_vertices, True, (255, 0, 0), 2)

        # Draw center points
        for point in center_points:
            cv2.circle(result, point, 5, (0, 255, 0), -1)

        return center_points, result


def adjust_roi(event, x, y, flags, param):
    """Mouse callback for adjusting ROI vertices."""
    detector, vertex_idx = param
    if event == cv2.EVENT_LBUTTONDOWN:
        # Update the selected vertex
        detector.roi_vertices[0][vertex_idx] = (x, y)
        detector.update_perspective_transform()


def update_white_threshold(value):
    """Callback for white threshold trackbar"""
    global lane_detector
    lane_detector.white_threshold = value
    lane_detector.save_thresholds()


def main():
    global lane_detector

    # Open webcam
    cap = cv2.VideoCapture(0)

    # Set lane width (in pixels, adjust based on your setup)
    lane_width = 300

    # Define initial region of interest
    roi_vertices = np.array([
        [(100, 480), (320, 300), (520, 300), (740, 480)]
    ], dtype=np.int32)

    # Initialize lane detector
    lane_detector = LaneDetector(lane_width, roi_vertices)

    # Create window for ROI adjustment
    cv2.namedWindow('ROI Adjustment')
    vertex_idx = 0
    cv2.setMouseCallback('ROI Adjustment', adjust_roi, (lane_detector, vertex_idx))

    # Create window for threshold adjustment
    cv2.namedWindow('Threshold Adjustment')
    cv2.createTrackbar('White Threshold', 'Threshold Adjustment',
                       lane_detector.white_threshold, 255, update_white_threshold)

    # Create a sample image to display in threshold window
    threshold_display = np.zeros((200, 400, 3), dtype=np.uint8)

    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            break

        # Create a copy for ROI adjustment
        roi_frame = frame.copy()

        # Draw ROI
        cv2.polylines(roi_frame, lane_detector.roi_vertices, True, (255, 0, 0), 2)

        # Draw vertices with labels
        for i, vertex in enumerate(lane_detector.roi_vertices[0]):
            color = (0, 255, 0) if i == vertex_idx else (0, 0, 255)
            cv2.circle(roi_frame, (vertex[0], vertex[1]), 5, color, -1)
            cv2.putText(roi_frame, str(i), (vertex[0] + 10, vertex[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Display ROI adjustment frame
        cv2.imshow('ROI Adjustment', roi_frame)

        # Update threshold display
        threshold_display[:] = 0  # Clear
        cv2.putText(threshold_display, f"White Threshold: {lane_detector.white_threshold}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(threshold_display, "Higher values = stricter white detection",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(threshold_display, "Lower values = more white detected",
                    (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow('Threshold Adjustment', threshold_display)

        # Handle key inputs
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            # Switch to next vertex
            vertex_idx = (vertex_idx + 1) % 4
            cv2.setMouseCallback('ROI Adjustment', adjust_roi, (lane_detector, vertex_idx))
        elif key == ord('s'):
            # Save thresholds
            lane_detector.save_thresholds()
            print("Thresholds saved")

        # Measure FPS
        start_time = time.time()

        # Detect lanes and get center points
        center_points, visualization_img = lane_detector.detect_lanes_and_center(frame)

        # Display result
        cv2.imshow('Lane Detection', visualization_img)

        # Calculate FPS
        fps = 1 / (time.time() - start_time)
        print(f"FPS: {fps:.2f}")

    # Save thresholds before exiting
    lane_detector.save_thresholds()

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()