import cv2
import os
import glob
import time


def play_tiff_video(folder_path, fps=30):
    """
    Play all .tiff images in a folder as a looping video

    Args:
        folder_path (str): Path to folder containing .tiff images
        fps (int): Frames per second for playback
    """

    # Get all .tiff files in the folder (case insensitive)
    tiff_pattern = os.path.join(folder_path, "*.tif*")
    image_files = glob.glob(tiff_pattern, recursive=False)

    # Also check for uppercase extensions
    tiff_pattern_upper = os.path.join(folder_path, "*.TIF*")
    image_files.extend(glob.glob(tiff_pattern_upper, recursive=False))

    if not image_files:
        print(f"No .tiff images found in {folder_path}")
        return

    # Sort files to ensure consistent order
    image_files.sort()

    print(f"Found {len(image_files)} .tiff images")
    print("Press 'q' to quit, 'p' to pause/resume")

    # Calculate delay between frames (in milliseconds)
    frame_delay = int(1000 / fps)

    paused = False

    try:
        while True:  # Loop forever
            for img_path in image_files:
                # Read image
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)

                if img is None:
                    print(f"Warning: Could not read {img_path}")
                    continue

                # Display image
                cv2.imshow('TIFF Video Player', img)

                # Handle key presses
                key = cv2.waitKey(frame_delay) & 0xFF

                if key == ord('q'):  # Quit
                    return
                elif key == ord('p'):  # Pause/Resume
                    paused = not paused
                    print("Paused" if paused else "Resumed")

                # If paused, wait for key press
                while paused:
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord('p'):
                        paused = False
                        print("Resumed")
                    elif key == ord('q'):
                        return

    except KeyboardInterrupt:
        print("\nStopped by user")

    finally:
        cv2.destroyAllWindows()


# Main execution
if __name__ == "__main__":
    # Specify your folder path here
    #folder_path = input("Enter the folder path containing .tiff images: ").strip()

    # Remove quotes if user copied path with quotes
    folder_path = "tiff_images"

    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
    else:
        play_tiff_video(folder_path, fps=30)