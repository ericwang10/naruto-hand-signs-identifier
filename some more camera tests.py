import numpy as np
import cv2
from datetime import datetime, time
from tensorflow import keras

cap = cv2.VideoCapture(0)

# cap.set(3,1280)
# cap.set(4,1024)
#1280x1024, 1024x768, 640x480, 352x288
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

backSub = cv2.createBackgroundSubtractorMOG2()
backSub = cv2.bgsegm.createBackgroundSubtractorMOG()

#data creation variables
time_elapsed = 0
start_time = True
sign_number = 0
destination = "test"
pathgray = "naruto handsign data/" + destination +"/"
pathRGB = "naruto handsign data RGB/" + destination +"/"
record = False
run_number = "#1"

signlist = ["bird","boar","dog","dragon","hare","horse","monkey","ox"]

rectx1 = 229 #to remove the fckin border
rectx2 = 531
recty1 = 249
recty2 = 551

#model prediction stuff
predict_mode = False
model = keras.models.load_model("D:/udemy/my model")

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
    fgMask = backSub.apply(frame, learningRate = 0)

    #flipping images
    frame = cv2.flip(frame,1)
    fgMask = cv2.flip(fgMask,1)

    cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv2.putText(frame, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    #turning mask into rgb
    mask_rgb = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2RGB)
    out_frame = cv2.bitwise_and(frame, mask_rgb)

    cv2.rectangle(frame, (rectx1, recty1), (rectx2, recty2), (255, 0, 0), 2)
    cv2.rectangle(fgMask, (rectx1, recty1), (rectx2, recty2), (255, 0, 0), 2)
    cv2.rectangle(out_frame, (rectx1, recty1), (rectx2, recty2), (255, 0, 0), 2)


    # start of data creation
    if record:
        if(start_time):
            start = datetime.now()
            start_time = False
        current_time = datetime.now()
        seconds_past = (current_time-start).seconds
        #print(seconds_past)

        sign = signlist[sign_number]

        cv2.putText(frame, "TIME REMAINING " +str(seconds_past) +" NEXT SIGN "+ sign, (100,100),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
        cv2.putText(fgMask, "TIME REMAINING " +str(seconds_past) +" NEXT SIGN "+ sign, (100,100),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
        cv2.putText(out_frame, "TIME REMAINING " +str(seconds_past) +" NEXT SIGN "+ sign, (100,100),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

        #print(fgMask[320:320,20:20])
        if(seconds_past==1): #3 whole seconds
            true_path_gray = pathgray + sign + "/" #this gives correct folder
            true_path_RGB = pathRGB + sign + "/"

            #true_path_gray = "naruto handsign data/"

            # this is fckin different from the rectangle coordinates, cuz its y1:y2, x1:x2
            cv2.imwrite(true_path_gray + sign + str(time_elapsed) + run_number + ".png", fgMask[recty1:recty2, rectx1:rectx2]) #gray
            cv2.imwrite(true_path_RGB + sign + str(time_elapsed) + run_number + ".png", out_frame[recty1:recty2, rectx1:rectx2]) #gray

            print(true_path_gray + sign + str(time_elapsed) + run_number + ".png")
            print(time_elapsed)
            #sign_number+=1
            if sign_number == len(signlist):
                sign_number = 0
            time_elapsed +=1
            start = datetime.now()
            if (time_elapsed==10):
                record = False  # pause
                start_time = True
                time_elapsed = 0
    #out_frame[recty1:recty2,rectx1:rectx2] |= frame[recty1:recty2, rectx1:rectx2] this just prevents the square from being absorbed, it just remains the same
    # end of data creation

    if (predict_mode):
        hand_frame = out_frame[recty1:recty2, rectx1:rectx2]
        hand_frame_resized = cv2.resize(hand_frame,(150,150))
        hand_frame_resized = np.expand_dims(hand_frame_resized, axis =0)
        decision = model.predict(hand_frame_resized)[0] #it randomly gives a fckin [[1,2,3]] array for some reason??? a freaking 2d array
        index = np.argmax(decision) #can't directly just plug the np.argmax as the index
        print(decision)
        if(decision[index]>0.9): # 90% confidence of ANY sign
            cv2.putText(frame, signlist[index] + str(decision[index]), (rectx1, recty1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            #cv2.putText(frame, np.array2string(decision,precision = 2), (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

        else:
            cv2.putText(frame, "sign unknown", (rectx1, recty1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Frame', frame)
    cv2.imshow("Foreground Mask",fgMask)
    cv2.imshow("FG COLOR", out_frame)
    k = cv2.waitKey(1)
    if k==ord('q'):
        break
    elif k==ord('r'):
        backSub=None
        backSub = cv2.createBackgroundSubtractorMOG2()
    elif k==ord('b'):
        backSub = cv2.bgsegm.createBackgroundSubtractorMOG()
    elif k == ord('p'):
        #print("predict")
        record = False #pause
        start_time = True
        time_elapsed = 0
    elif k == ord('n'): #record data
        record = True
    elif k == ord('m'): #next sign
        sign_number += 1
    elif k == ord('w'): #weave
        predict_mode = True


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()