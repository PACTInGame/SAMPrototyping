import cv2

cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Fehler beim Öffnen der Kamera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kein Bild erhalten!")
        break
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == 27:  # Escape zum Beenden
        break

cap.release()
cv2.destroyAllWindows()