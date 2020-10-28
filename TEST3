# USAGE
# python text_detection_video.py --east frozen_east_text_detection.pb

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import pytesseract
from googletrans import Translator

pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\pc\\Desktop\\Tesseract-OCR\\tesseract'
translator = Translator()

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
                help="path to optinal input video file")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
                help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=640,
                help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=640,
                help="resized image height (should be multiple of 32)")
ap.add_argument("-l", "--lang", default="eng",
                help="language that Tesseract will use when OCR'ing")
ap.add_argument("-p", "--psm", type=int, default=6,
                help="Tesseract PSM mode")
args = vars(ap.parse_args())

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])

# start the FPS throughput estimator
fps = FPS().start()

# loop over frames from the video stream
while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    # check to see if we have reached the end of the stream
    if frame is None:
        break

    # resize the frame, maintaining the aspect ratio
    frame = imutils.resize(frame, width=1000)
    image = frame.copy()
    cv2.imshow("Video", image)

    key1 = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key1 == ord("q"):
        frame = vs.read()
        cv2.destroyAllWindows()

        # re-recognize the text from the frame
        image = frame.copy()
        img_raw = image.copy()
        ROIs = cv2.selectROIs("Select Rois", img_raw)
        # print rectangle points of selected roi
        print(ROIs)
        # loop over every bounding box save in array "ROIs"
        for rect in ROIs:
            x1 = rect[0]
            y1 = rect[1]
            x2 = rect[2]
            y2 = rect[3]

            # crop roi from original image
            img_crop = img_raw[y1:y1 + y2, x1:x1 + x2]

            # # show cropped image
            # cv2.imshow("crop" + str(crop_number), img_crop)
            cv2.destroyAllWindows()
            gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
            options = "-l {} --psm {}".format(args["lang"], args["psm"])
            text = pytesseract.image_to_string(gray, config=options)
            text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
            # show the original OCR'd text
            print("ORIGINAL")
            print("========")
            print(text)
            print("")
            textTranslate = translator.translate(text, src='en', dest='fr')
            print('Translation')
            print(f'source: {textTranslate.src}')
            print(f'Destination: {textTranslate.dest}')
            print(f'{textTranslate.origin} -> {textTranslate.text}')
            print("")
            cv2.rectangle(img_raw, (x1, y1), (x1 + x2, y1 + y2), (0, 0, 255), 2)
            cv2.putText(img_raw, textTranslate.text, (x1 - 5, y1), cv2.FONT_HERSHEY_COMPLEX,
                        0.75, (255, 0, 0), 2)
        cv2.imshow("Text Detection", img_raw)

        cv2.waitKey(0)
    key2 = cv2.waitKey(1) & 0xFF
    if key2 == ord("z"):
        break
cv2.destroyAllWindows()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# if we are using a webcam, release the pointer
if not args.get("video", False):
    vs.stop()

# otherwise, release the file pointer
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()
