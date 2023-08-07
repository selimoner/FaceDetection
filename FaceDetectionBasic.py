import cv2
import mediapipe as mp
import time

# cap = cv2.VideoCapture("Videos/7.mp4")
cap = cv2.VideoCapture(0)

width = int(cap.get(3))
height = int(cap.get(4))

print(f"Width : {width}, Height : {height}")

if height > 1100:
    height=height//3
if width > 2000:
    width=width//3

previousTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

while True:

    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    # print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box

            image_height, image_width, image_channel = img.shape

            bbox = int(bboxC.xmin * image_width), int(bboxC.ymin * image_height), int(bboxC.width * image_width), int(bboxC.height * image_height)

            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    img = cv2.resize(img, (width,height))

    cv2.putText(img, f"FPS : {int(fps)}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    cv2.imshow("IMG", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break