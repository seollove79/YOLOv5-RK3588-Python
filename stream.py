import cv2

gst_str = ('rtspsrc location=rtsp://192.168.144.108:554/stream=1 ! '
           'rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink')
cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 여기서 frame을 처리합니다
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()