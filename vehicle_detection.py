import numpy as np
import cv2 as cv


car_cascade = cv.CascadeClassifier("E:\FYP\haar-cascades\cars6.xml")
object_detector = cv.createBackgroundSubtractorMOG2()

cap = cv.VideoCapture(r"E:\FYP\videos\video2.avi")
count = 0
kernel = np.ones((5,5),np.uint8)



while True:
    ret, frame = cap.read()
    frame = cv.resize(frame, (720,480))
    height, width = frame.shape[0:2]
    frame[0:50,0:width] = [25,9,51]
    cv.putText(frame,'Vehicle Count: ', (10,40), cv.FONT_HERSHEY_TRIPLEX, 1.5, (255,255,255), 2)
    cv.line(frame,(0,height-400), (width, height-400), (0,255,0), 2)
    cv.line(frame,(0,height-200), (width, height-200), (255,0,0), 2)

    mask = object_detector.apply(frame)
    erosion = cv.erode(mask,kernel,iterations = 1)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray,1.3, 5)
    
    for (x,y,w,h) in cars:
        carCy = int(y+h/2)
        lineCy = height-400

        if (carCy<lineCy+6 and carCy>lineCy-6):
            count  = count+1
            cv.line(frame,(0,height-200), (width, height-200), (60,255,60), 2)


        cv.rectangle(frame, (x,y), (x+w, y+h), (222,107,0), 2)
        cv.putText(frame, "Car", (x,y-10),  cv.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
        cv.putText(frame, str(count), (500,30), cv.FONT_HERSHEY_COMPLEX, 1.5, (255,255,255), 2)
    

    cv.imshow("Detection", frame)
    cv.imshow("Threshold", mask)
    cv.imshow("Morphological Analysis", erosion)
    k = cv.waitKey(100) & 0xFF
    if k ==97:
        break

cap.release()
cv.destroyAllWindows()