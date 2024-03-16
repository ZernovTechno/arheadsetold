import cv2
import mediapipe as mp
from flask_opencv_streamer.streamer import Streamer
import matplotlib.pyplot as plt
from math import asin, degrees, sqrt
from PIL import Image
from PIL import ImageOps
import imutils
import numpy as np
import datetime
import imagezmq
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from pyzbar import pyzbar

segmentor = SelfiSegmentation()

k_1 = 0.05
k_2 = 0.05

class BarrelDeformer:

    def transform(self, x, y):
        # center and scale the grid for radius calculation (distance from center of image)
        x_c, y_c = 1024 / 2, 1024 / 2 
        x = (x - x_c) / x_c
        y = (y - y_c) / y_c
        radius = np.sqrt(x**2 + y**2) # distance from the center of image
        m_r = 1 + k_1*radius + k_2*radius**2 # radial distortion model
        # apply the model 
        x, y = x * m_r, y * m_r
        # reset all the shifting
        x, y = x*x_c + x_c, y*y_c + y_c
        return x, y

    def transform_rectangle(self, x0, y0, x1, y1):
        return (*self.transform(x0, y0),
                *self.transform(x0, y1),
                *self.transform(x1, y1),
                *self.transform(x1, y0),
                )

    def getmesh(self, img):
        self.w, self.h = img.size
        gridspace = 20
        target_grid = []
        for x in range(0, self.w, gridspace):
            for y in range(0, self.h, gridspace):
                target_grid.append((x, y, x + gridspace, y + gridspace))
        source_grid = [self.transform_rectangle(*rect) for rect in target_grid]
        return [t for t in zip(target_grid, source_grid)]

def removebackgrnd(capture):

    result2 = segmentor.removeBG(capture, (0, 0, 0), 0.3)
    src = result2
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)
    return dst

class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplex=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplex
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)

        self.mpDraw = mp.solutions.drawing_utils
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(self.results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img
    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmList
    
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60) # Частота кадров
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # Ширина кадров в видеопотоке.
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # Высота кадров в видеопотоке.
    detector = handDetector()
    iteration = 0
    button1 = False

    lastfingerpos = [0,0]
    lastfingeriter = 0

    timer = 0

    clockssize = [200, 100]
    clockspoint = [50, 50]
    clocks = True

    image_hub = imagezmq.ImageHub()
    while True:
        ret, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        #img = cv2.rectangle(img, (0,0), (1280,720), (255, 255, 255), -1)
        cropimg = img.copy()
        if button1 == False:
                        if (clocks == True):
                            cv2.rectangle(img, (clockspoint[0]-5, clockspoint[1]-5), (clockspoint[0]+clockssize[0]+5, clockspoint[1]+clockssize[1]+5), (0, 0, 0), -1, )
                            cv2.rectangle(img, (clockspoint[0], clockspoint[1]), (clockspoint[0]+clockssize[0], clockspoint[1]+clockssize[1]), (255, 255, 255), -1, )

                            cv2.line(img, (clockspoint[0]+clockssize[0]-15, clockspoint[1]+10), (clockspoint[0]+clockssize[0]-30, clockspoint[1]+10), (0, 0, 0), 3)

                            now = datetime.datetime.now()
                            time = now.strftime("%H:%M")
                            data = now.strftime("%d-%m-%Y")
                            cv2.putText(img, data, (clockspoint[0] + clockssize[0] //6, clockspoint[1] + clockssize[1]//2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
                            cv2.putText(img, time, (clockspoint[0] + clockssize[0] //4 + 2, clockspoint[1] + clockssize[1]//2+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                        else:
                            cv2.circle(img, (1280//2, 720 - 60), 55, (0, 0, 0), -1)
                            cv2.circle(img, (1280//2, 720 - 60), 50, (255, 255, 255), -1)
                            cv2.line(img, (1280//2, 720 - 60), ((1280//2-5, 720 - 60-30)), (0,0,0), 3)
                            cv2.line(img, (1280//2, 720 - 60), ((1280//2+20, 720 - 60-15)), (0,0,0), 5)
                            cv2.line(img, (1280//2-2, 720 - 60), ((1280//2+5, 720 - 60+35)), (0,0,0), 1)


        color_coverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        pil_image = Image.fromarray(color_coverted) 
        if (len(lmList) >= 20):
            if (abs(lmList[8][2] - lmList[4][2]) + abs(lmList[8][1] - lmList[4][1]) <= 50):
                cv2.rectangle(img, (1280 - 150, 720 - 150), (1280 - 50, 720 - 50), (255, 255, 255), 3)
            else:
                cv2.rectangle(img, (1280 - 150, 720 - 150), (1280 - 50, 720 - 50), (255, 0, 0), 3)
            xlist = []
            ylist = []

            for ids in lmList:
                xlist.append(ids[1])
                ylist.append(ids[2])

            minx = min(xlist) - 40
            maxx = max(xlist) + 40
            miny = min(ylist) - 40
            maxy = max(ylist) + 40
            if (minx <= 0): minx = 1
            if (maxx <= 0): maxx = 1
            if (maxy <= 0): maxy = 1
            if (miny <= 0): miny = 1

            cropimg = cropimg[miny:maxy, minx:maxx]
            cropimgwithoutbg = removebackgrnd(cropimg)

            if (clocks == True):
                if button1 == True:
                        cv2.rectangle(img, (lmList[8][1] - clockssize[0]//2 -5, lmList[8][2] -5), (lmList[8][1] + clockssize[0]//2 +5, lmList[8][2] + clockssize[1] +5), (0, 0, 0), -1, )
                        cv2.rectangle(img, (lmList[8][1] - clockssize[0]//2, lmList[8][2]), (lmList[8][1] + clockssize[0]//2, lmList[8][2] + clockssize[1]), (255, 255, 255), -1, )

                        now = datetime.datetime.now()
                        time = now.strftime("%H:%M")
                        data = now.strftime("%d-%m-%Y")
                        cv2.putText(img, data, (lmList[8][1] - clockssize[0]//2 + clockssize[0] //6, lmList[8][2] + clockssize[1]//2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
                        cv2.putText(img, time, (lmList[8][1] - clockssize[0]//2 + clockssize[0] //4 + 2, lmList[8][2] + clockssize[1]//2+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  
                        if (timer >= 5):
                            if (lmList[8][1] - lastfingerpos[0] <= 5 and lmList[8][2] - lastfingerpos[1] <= 5):
                                timer = 0
                                lastfingeriter += 1
                                if (lastfingeriter >= 5):
                                  lastfingeriter = 0
                                  button1 = False
                                  clockspoint = [lmList[8][1]-clockssize[0]//2, lmList[8][2]]
                            elif (lastfingeriter >= 3): lastfingeriter -= 3
                        print(lastfingeriter)
                elif (lmList[8][1] <= clockspoint[0]+clockssize[0] and lmList[8][1] >= clockspoint[0]+clockssize[0]-50 and lmList[8][2] >= clockspoint[1] and lmList[8][2] <= clockspoint[1]+25):
                    if (timer > 5):
                        timer = 0
                    elif (timer == 5):
                        clocks = False      
                        button1 = False
                elif (lmList[8][1] <= clockspoint[0] + clockssize[0] and lmList[8][1] >= clockspoint[0] and lmList[8][2] <= clockspoint[1] + clockssize[1] and lmList[8][2] >= clockspoint[1]):
                    if (False == True):
                        pass
                    else:
                        iteration += 1
                        print(iteration)
                        if (iteration >= 20):
                            button1 = not(button1)
                            iteration = 0
                else:
                    if ((iteration - 5) < 1): iteration = 0
                    else: iteration -= 5
            elif clocks == False:
                if (lmList[8][1] <= 1280/2+25 and lmList[8][1] >= 1280/2-25 and lmList[8][2] <= 720-10 and lmList[8][2] >= 720-110):
                    if (timer > 5):
                        timer = 0
                    elif (timer == 5):
                        clocks = True      


            lastfingerpos[0] = lmList[8][1]
            lastfingerpos[1] = lmList[8][2]

            timer += 1
            
            cropinpillow = (Image.fromarray(cv2.cvtColor(cropimgwithoutbg, cv2.COLOR_BGRA2RGBA)))
            cropinpillow = cropinpillow.convert("RGBA")

            color_coverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA) 
            pil_image = Image.fromarray(color_coverted) 
            pil_image.paste(cropinpillow, (minx,miny), cropinpillow)


        pil_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGRA2RGBA)

        #rpi_name, imager = image_hub.recv_image()
        #image_hub.send_reply(b'OK')  # this statement is missing from your while True loop

        #display = (Image.fromarray(cv2.cvtColor(imager, cv2.COLOR_BGRA2RGBA)))

        pil_image = Image.fromarray(cv2.cvtColor(pil_image, cv2.COLOR_BGRA2RGBA))
        #pil_image.paste(display, (0,212), display)  

        #pil_image = ImageOps.deform(pil_image, BarrelDeformer())

        pil_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGRA2RGBA)

        print(pyzbar.decode(pil_image))

        pil_image = cvzone.stackImages([pil_image, pil_image], 2, 0.7)
        cv2.imshow("Image", pil_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
if __name__ == "__main__":
    main()