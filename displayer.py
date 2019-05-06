import cv2
import matplotlib.pyplot as plt
import numpy as np
import csv
import plotly
import plotly.graph_objs as go
import plotly.io as pio
import io
from scipy.fftpack import fftfreq, fft
from PIL import *

from plotly.offline import *

filename = 'RC_VID_0020.mp4'
cap = cv2.VideoCapture(filename)

# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video  file")

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

gyrox = []
gyroy = []
gyroz = []
motor0 = []
motor1 = []
motor2 = []
motor3 = []
contr0 = []
contr1 = []
contr2 = []
contr3 = []
head0 = []
head1 = []
head2 = []

gyroFFT = 0
motorFFT = 0
contrFFT = 0
headFFT = 0

blackBoxCount = 0
currentPosition = 0

with open('20190505.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    for i in range(82):
        next(csv_reader)

    for row in csv_reader:
        contr0.append(row[11])
        contr1.append(row[12])
        contr2.append(row[13])
        contr3.append(row[14])
        gyrox.append(row[17])
        gyroy.append(row[18])
        gyroz.append(row[19])
        motor0.append(row[23])
        motor1.append(row[24])
        motor2.append(row[25])
        motor3.append(row[26])
        head0.append(row[32])
        head1.append(row[33])
        head2.append(row[34])
        blackBoxCount += 1

def mapInput(frameNum):
    return round((frameNum / (length)) * blackBoxCount)

def makeData(arr, names):
    data = []

    for i in range(len(arr)):
        data.append(go.Scatter(
            y = arr[i],
            mode = 'lines',
            name = names[i]
        ))

    return data

def adjustRange(length, frame, arr, skip):
    startFrame = frame - length * 30
    endFrame = frame

    if startFrame < 0:
        startFrame = 0

    datastart = mapInput(startFrame)
    dataend = mapInput(endFrame)

    for i in range(len(arr)):
        arr[i] = arr[i][datastart:dataend:skip]

    return arr

def generateImage(arr,names, title):
    arr = makeData(arr,names)
    layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        width=1920/2,
        height=1080/2,
        title=title
    )
    #plotly.offline.plot(adjustRange(2, 500, [gyrox, gyroy, gyroz]), filename='line-mode.html')

    img_bytes = pio.to_image(dict(data=arr, layout=layout), format='png')
    return img_bytes

def generateFFTImage(arr,names, title):
    for i in range(len(arr)):
        if(len(arr[i]) > 20):
            val = np.abs(fft(arr[i]))

            mid=round((len(val) + 1) / 2)
            arr[i] = val[:mid]

    return generateImage(arr,names, title)



def onChange(trackbarValue):
    global currentPosition
    currentPosition = trackbarValue
    cap.set(cv2.CAP_PROP_POS_FRAMES,trackbarValue)
    err,img = cap.read()

    if gyroFFT == 0:
        gyro = generateImage(adjustRange(2, trackbarValue, [gyrox, gyroy, gyroz], 10), ["Gyro X", "Gyro Y", "Gyro Z"], "Gyroscope Data 2 Seconds")
    elif gyroFFT == 1:
        gyro = generateFFTImage(adjustRange(2, trackbarValue, [gyrox, gyroy, gyroz], 10), ["Gyro X", "Gyro Y", "Gyro Z"], "Gyroscope FFT 2 Seconds")
    elif gyroFFT == 2:
        gyro = generateImage(adjustRange(20, trackbarValue, [gyrox, gyroy, gyroz], 20), ["Gyro X", "Gyro Y", "Gyro Z"], "Gyroscope Data 20 Seconds")
    elif gyroFFT == 3:
        gyro = generateFFTImage(adjustRange(20, trackbarValue, [gyrox, gyroy, gyroz], 20), ["Gyro X", "Gyro Y", "Gyro Z"], "Gyroscope FFT 20 Seconds")

    if motorFFT == 0:
        motor = generateImage(adjustRange(2, trackbarValue, [motor0, motor1, motor2, motor3], 10), ["Motor 0", "Motor 1", "Motor 2", "Motor 3"], "Motor Data 2 Seconds")
    elif motorFFT == 1:
        motor = generateFFTImage(adjustRange(2, trackbarValue, [motor0, motor1, motor2, motor3], 10), ["Motor 0", "Motor 1", "Motor 2", "Motor 3"], "Motor FFT 2 Seconds")
    elif motorFFT == 2:
        motor = generateImage(adjustRange(20, trackbarValue, [motor0, motor1, motor2, motor3], 20), ["Motor 0", "Motor 1", "Motor 2", "Motor 3"], "Motor Data 20 Seconds")
    elif motorFFT == 3:
        motor = generateFFTImage(adjustRange(20, trackbarValue, [motor0, motor1, motor2, motor3], 20), ["Motor 0", "Motor 1", "Motor 2", "Motor 3"], "Motor FFT 20 Seconds")

    if contrFFT == 0:
        contr = generateImage(adjustRange(2, trackbarValue, [contr0, contr1, contr2, contr3], 10), ["Control 0", "Control 1", "Control 2", "Control 3"], "Control Data 2 Seconds")
    elif contrFFT == 1:
        contr = generateFFTImage(adjustRange(2, trackbarValue, [contr0, contr1, contr2, contr3], 10), ["Control 0", "Control 1", "Control 2", "Control 3"], "Control FFT 2 Seconds")
    elif contrFFT == 2:
        contr = generateImage(adjustRange(20, trackbarValue, [contr0, contr1, contr2, contr3], 20), ["Control 0", "Control 1", "Control 2", "Control 3"], "Control Data 20 Seconds")
    elif contrFFT == 3:
        contr = generateFFTImage(adjustRange(20, trackbarValue, [contr0, contr1, contr2, contr3], 20), ["Control 0", "Control 1", "Control 2", "Control 3"], "Control FFT 20 Seconds")

    if headFFT == 0:
        head = generateImage(adjustRange(2, trackbarValue, [head0, head1, head2], 10), ["Heading 0", "Heading 1", "Heading 2"], "Heading Data 2 Seconds")
    elif headFFT == 1:
        head = generateFFTImage(adjustRange(2, trackbarValue, [head0, head1, head2], 10), ["Heading 0", "Heading 1", "Heading 2"], "Heading FFT 2 Seconds")
    elif headFFT == 2:
        head = generateImage(adjustRange(20, trackbarValue, [head0, head1, head2], 20), ["Heading 0", "Heading 1", "Heading 2"], "Heading Data 20 Seconds")
    elif headFFT == 3:
        head = generateFFTImage(adjustRange(20, trackbarValue, [head0, head1, head2], 20), ["Heading 0", "Heading 1", "Heading 2"], "Heading FFT 20 Seconds")


    gyronp = cv2.imdecode(np.frombuffer(gyro, np.uint8), cv2.IMREAD_COLOR)
    motornp = cv2.imdecode(np.frombuffer(motor, np.uint8), cv2.IMREAD_COLOR)
    contrnp = cv2.imdecode(np.frombuffer(contr, np.uint8), cv2.IMREAD_COLOR)
    headnp = cv2.imdecode(np.frombuffer(head, np.uint8), cv2.IMREAD_COLOR)

    gh1, gw1 = gyronp.shape[:2]
    mh1, mw1 = motornp.shape[:2]
    ch1, cw1 = contrnp.shape[:2]
    hh1, hw1 = headnp.shape[:2]

    vis = np.zeros((gh1 + ch1, gw1 + mw1, 3), np.uint8)

    vis[:gh1, :gw1] = gyronp
    vis[:mh1, gw1:gw1+mw1] = motornp
    vis[gh1:gh1+ch1, :cw1] = contrnp
    vis[gh1:gh1+ch1, gw1:gw1+mw1] = headnp

    added_image = cv2.addWeighted(img,0.6,vis,1,0)

    resize = cv2.resize(added_image, (1920, 1020))
    cv2.imshow("Blackbox Viewer", resize)
    pass

def checkPos(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global gyroFFT, motorFFT, contrFFT, headFFT
        if x < 960:
            if y < 540:
                gyroFFT += 1
                if gyroFFT > 3:
                    gyroFFT = 0
            else:
                contrFFT += 1
                if contrFFT > 3:
                    contrFFT = 0
        else:
            if y < 540:
                motorFFT += 1
                if motorFFT > 3:
                    motorFFT = 0
            else:
                headFFT += 1
                if headFFT > 3:
                    headFFT = 0

        onChange(currentPosition)
    pass


cv2.namedWindow('Blackbox Viewer')
cv2.createTrackbar( 'Start', 'Blackbox Viewer', 0, length - 1, onChange)
cv2.setMouseCallback('Blackbox Viewer',checkPos)

onChange(0)
cv2.waitKey()

start = cv2.getTrackbarPos('start','Blackbox Viewer')

cap.set(cv2.CAP_PROP_POS_FRAMES,start)

while(cap.isOpened()):

  ret, frame = cap.read()

  if ret == True:
    resize = cv2.resize(frame, (1920, 1020))
    cv2.imshow('Blackbox Viewer', resize)

    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  else:
    break

cap.release()

cv2.destroyAllWindows()
