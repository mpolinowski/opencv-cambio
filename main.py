import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np
import os

RTSP_URL = 'rtsp://admin:instar@192.168.2.120/livestream/12'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

fpsReader = cvzone.FPS()

# get RTSP stream
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(3, 1920)
cap.set(4, 1080)

# create instance of colorfinder
# set True to display sliders to adjust search colour
cvColorFinder = ColorFinder(False)
# colour to search for
hsvVals = {'hmin': 0, 'smin': 0, 'vmin': 0, 'hmax': 179, 'smax': 142, 'vmax': 176}

if not cap.isOpened():
    print('ERROR :: Cannot open RTSP stream')
    exit(-1)

def empty(a):
    pass 

# create sliders to adjust
# settings on the fly
cv2.namedWindow("Settings")
cv2.resizeWindow("Settings", 640, 240)
cv2.createTrackbar("Canny Threshold1", "Settings", 55, 255, empty)
cv2.createTrackbar("Canny Threshold2", "Settings", 60, 255, empty)
cv2.createTrackbar("CV Min Area", "Settings", 40000, 100000, empty)

# prepare the image for detection
def preProcessing(img):

    # add some blur to reduce noise
    img_prep = cv2.GaussianBlur(img, (5, 5), 3)
    # use canny filter to enhance contours
    # make thresholds changeable by sliders
    threshold1 = cv2.getTrackbarPos("Canny Threshold1", "Settings")
    threshold2 = cv2.getTrackbarPos("Canny Threshold2", "Settings")
    img_prep = cv2.Canny(img_prep, threshold1, threshold2)
    # make features more prominent by dilations
    kernel = np.ones((5, 5), np.uint8)
    img_prep = cv2.dilate(img_prep, kernel, iterations=1)
    # morph detected features to close gaps in geometries
    img_prep = cv2.morphologyEx(img_prep, cv2.MORPH_CLOSE, kernel)

    return img_prep

# while the stream runs do detection
while True:
    success, img = cap.read()
    # show fps counter
    # fps, img = fpsReader.update(img,pos=(50,80),color=(0,255,0),scale=5,thickness=5)
    # pre-process each image
    img_prep = preProcessing(img)
    # min area slider to filter noise
    cvMinArea = cv2.getTrackbarPos("CV Min Area", "Settings")
    # findContours returns the processed image and found contours
    imgContours, conFound = cvzone.findContours(img, img_prep, cvMinArea)
    # conFound will contain all contours found
    # we can limit it to circles for our coins
    moneyCountByContour = 0
    moneyCountByColour = 0
    if conFound:
        for contour in conFound:
            # get the arc length of the contour
            perimeter = cv2.arcLength(contour["cnt"], True)
            # calculate approx polygon count / corner points
            polycount = cv2.approxPolyDP(contour["cnt"], 0.02 * perimeter, True)
            # print # of corner points in contour
            # print(len(polycount))

            if len(polycount) >= 8:

                # GET AREA BY CONTOUR AREA
                area = contour['area']
                # print(area)


                if 43000 < area < 49000:
                    moneyCountByContour += .1

                elif 49000 < area < 55000:
                    moneyCountByContour += .2

                elif 55000 < area < 72000:
                    moneyCountByContour += .5

                elif 72000 < area < 104000:
                    moneyCountByContour += 1

                elif 104000 < area < 116000:
                    moneyCountByContour += 2

                elif 116000 < area < 120000:
                    moneyCountByContour += 5

                else:
                    moneyCountByContour += 0

                # GET AREA BY OBJECT COLOUR
                ## get location of bounding box
                x, y, w, h = contour['bbox']
                ## crop to bounding box
                imgCrop = img[y:y+h, x:x+w]
                ## show cropped image
                ## cv2.imshow('Cropped Contour', imgCrop)

                ## find colour based on hsvVals in imgCrop
                imgColour, mask = cvColorFinder.update(imgCrop, hsvVals)
                ## we adjusted the hsvVals that everything but the coins
                ## are black. Now we can exclude everything that is black
                ## and count the pixels that match our coin colour to
                ## get it's surface area.
                colouredArea = cv2.countNonZero(mask)
                print(colouredArea)


                if 39000 < colouredArea < 51000:
                    moneyCountByColour += .1

                elif 51000 < colouredArea < 55000:
                    moneyCountByColour += .2

                elif 59000 < colouredArea < 72000:
                    moneyCountByColour += .5

                elif 72000 < colouredArea < 98000:
                    moneyCountByColour += 1

                elif 100000 < colouredArea < 110000:
                    moneyCountByColour += 2

                elif 110000 < colouredArea < 112000:
                    moneyCountByColour += 5

                else:
                    moneyCountByColour += 0

    # print('Contour: ', moneyCountByContour)
    # print('Colour: ', moneyCountByColour)



    # show original vs pre-processed image
    # show all streams in 2 columns at 1/3 size
    imageStack = cvzone.stackImages([img, img_prep, imgContours], 2, 0.3)
    # add money counter
    cvzone.putTextRect(img=imageStack, text=f'{moneyCountByContour} HK$ (by Contour)', pos=(20, 50), thickness=2, colorR=(204,119,0))
    cvzone.putTextRect(img=imageStack, text=f'{moneyCountByColour} HK$ (by Colour)', pos=(20, 150), thickness=2, colorR=(204,119,0))
    # and show results
    cv2.imshow(RTSP_URL, imageStack)

    # keep running until you press `q`
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break