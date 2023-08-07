import cv2
import mediapipe as mp
import time

class FaceDetector():

    # mediapipe kütüphanesini tanımlıyoruz

    def __init__(self, minDetectionConfidence = 0.5):

        self.minDetectionConfidence = minDetectionConfidence

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionConfidence)

    # Görüntüde yüzleri bulan ve gösteren metodumuz

    def findFaces(self, img, draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results)

        boundingBoxes = [] # Bulduğumuz yüzleri imajlarıyla beraber bu listede topluyoruz

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):

                bboxC = detection.location_data.relative_bounding_box

                image_height, image_width, image_channel = img.shape

                bbox = int(bboxC.xmin * image_width), int(bboxC.ymin * image_height), int(
                    bboxC.width * image_width), int(bboxC.height * image_height)

                boundingBoxes.append([id, bbox, detection.score])

                if draw:
                    img = self.drawVersion2(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 255), 2)

        return img, boundingBoxes

    # Tespit edilen yüzü çevreleyen dikdörtgenlerin köşelerini süslüyoruz :)

    def drawVersion2(self, img, boundingBox, l = 30, t=5, rectangle_thickness = 1):
        x, y, width, height = boundingBox
        x1, y1 = x + width, y+height

        cv2.rectangle(img, boundingBox, (255, 0, 255), rectangle_thickness)

        # Sol üst
        cv2.line(img, (x,y), (x+l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)

        # Sağ üst
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y+l), (255, 0, 255), t)

        # Sol alt
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)

        # Sağ alt
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)

        return img

def main():

    # cap = cv2.VideoCapture("Videos/1.mp4") # Video tespit
    cap = cv2.VideoCapture(0) # Kameradan tespit

    # Burada videonun boyutunu ayarlıyorum, boyutu büyük olan videolar çalıştırılınca ekrandan taşıyor.

    width = int(cap.get(3))
    height = int(cap.get(4))

    # print(f"Width : {width}, Height : {height}") # Test amaçlı boyutunu konsola yazdırıyorum

    if height > 1100:
        height = height // 3
    if width > 2000:
        width = width // 3

    # -------------------------------------------------------------------------------------------------

    previousTime = 0

    detector = FaceDetector()

    while True:
        success, img = cap.read()
        img_with_boxes, boundingBoxes = detector.findFaces(img)  # Unpack the tuple

        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        img_with_boxes = cv2.resize(img_with_boxes, (width, height))  # Resize the image with boxes

        cv2.putText(img_with_boxes, f"FPS : {int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

        cv2.imshow("IMG", img_with_boxes)  # Show the resized image

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()