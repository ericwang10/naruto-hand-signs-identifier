import numpy as np
import cv2

cap = cv2.VideoCapture(0)

backSub = cv2.createBackgroundSubtractorMOG2()
backSub = cv2.bgsegm.createBackgroundSubtractorMOG()

while(True):
    # # Capture frame-by-frame
    # ret, frame = cap.read()
    #
    # # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    # # Display the resulting frame
    # cv2.imshow('frame',gray)

    ret, frame = cap.read()

    fgMask = backSub.apply(frame, learningRate = 0.001)

    cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv2.putText(frame, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()