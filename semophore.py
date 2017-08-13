import cv2
import matplotlib
import numpy

trafficlightCascade = cv2.CascadeClassifier('cascade1.xml')
cap = cv2.VideoCapture('test.mp4')

while (cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    lights = trafficlightCascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in lights:
        cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('original', frame)
    cv2.imshow('gray', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
