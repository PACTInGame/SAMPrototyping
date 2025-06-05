import cv2

def find_available_cameras(max_index=10):
    available_cameras = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap is not None and cap.isOpened():
            available_cameras.append(index)
            cap.release()
    return available_cameras

print(find_available_cameras(10))