import argparse
import numpy as np
import cv2
import os
import mediapipe as mp
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import signal

parser = argparse.ArgumentParser()
parser.add_argument("-i","--input_video",type=str,help="the video path",required=True)
args = parser.parse_args()

def hand_detection(cap):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    #movement of two hands
    hands = mp_hands.Hands(static_image_mode=True,max_num_hands=2,min_detection_confidence=0.5)

    record_left = []
    record_right =[]
    img_lost = []
    right_lost=[]
    left_lost = []
    frame_count = 0
    img_list = []
    results = 0
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            # Convert the BGR image to RGB, flip the image around y-axis for correct 
            # handedness output and process it with MediaPipe Hands.
            results = hands.process(cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1))

            annotated_image = frame.copy()

            if results.multi_hand_landmarks != None:
                annotated_image = cv2.flip(frame.copy(), 1)
                right_hand = None
                left_hand = None

                if (len(results.multi_hand_landmarks) == 1 and results.multi_handedness[0].classification[0].label == "Left"):
                    right_hand = results.multi_hand_landmarks[0]
                    left_lost.append(frame_count)
                elif (len(results.multi_hand_landmarks) == 1 and results.multi_handedness[0].classification[0].label == "Right"):
                    left_hand = results.multi_hand_landmarks[0]
                    right_lost.append(frame_count)
                
                elif (len(results.multi_hand_landmarks) == 2):
                    hand0 = results.multi_hand_landmarks[0]
                    hand1 = results.multi_hand_landmarks[1]
                    #two same hands
                    if(results.multi_handedness[0].classification[0].label == results.multi_handedness[1].classification[0].label):
                        hand0_x = hand0.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
                        hand1_x = hand1.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x

                        left_hand = hand0 if hand0_x > hand1_x else hand1
                        right_hand = hand1 if hand0_x > hand1_x else hand0

                    else:
                        right_hand = hand0 if results.multi_handedness[0].classification[0].label == "Left" else hand1
                        left_hand = hand0 if results.multi_handedness[0].classification[0].label == "Right" else hand1
                
                if(right_hand != None):
                    mp_drawing.draw_landmarks(annotated_image, right_hand, mp_hands.HAND_CONNECTIONS)
                    record_right.append(right_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y)
                    #print(frame_count,"right", right_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x)
                if(left_hand != None):
                    mp_drawing.draw_landmarks(annotated_image, left_hand, mp_hands.HAND_CONNECTIONS)
                    record_left.append(left_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y)
                    #print(frame_count,"left", left_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x)

                #after processing the hand, flip back the image  
                annotated_image = cv2.flip(annotated_image, 1)
            else:
                img_lost.append(frame_count)

            img_list.append(annotated_image)
            frame_count += 1

        else:
            break

    cap.release()
    return img_list,record_right,record_left,right_lost,left_lost,img_lost
    
def find_peak(record_right, record_left):
    #multiple hand, 0 for right hand and 1 for left hand
    fist_closing_frame = []
    action_time_list = []
    fig, axs = plt.subplots(2)

    for hand in range(2):
        smooth = []
        label = ""
        if hand == 0:
            smooth = savgol_filter(record_right,9,3)
            label = "right hand"
        else:
            smooth = savgol_filter(record_left,9,3)
            label = "left hand"

        middle = (max(smooth)+min(smooth))/2
        pro = (max(smooth) - middle) * 0.8

        peaks, _ = signal.find_peaks(smooth,height=middle,prominence=pro)

        fist_closing_frame.append(peaks)

        #calculcate fist closing action time
        temp = []
        temp.append(peaks[0])
        for time in np.diff(peaks):
            temp.append(time)
        action_time_list.append(temp)

        axs[hand].plot(smooth) 
        axs[hand].plot(peaks,np.array(smooth)[peaks],"x")
        axs[hand].title.set_text(label)

    fig.tight_layout()
    fig.savefig('../demo_result/hand_movement.png')

    return fist_closing_frame, action_time_list


def demo():
    if(not os.path.isfile(args.input_video)):
        print("The input video doesn't exist!")
        return

    # Read video with OpenCV.
    cap=cv2.VideoCapture(args.input_video)

    ## Get video info
    RES=(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = round(cap.get(cv2.CAP_PROP_FPS))

    #create video writer to write detected_video
    output_video = "../demo_result/hand_movement_result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, RES)

    img_list,record_right,record_left,right_lost,left_lost, img_lost = hand_detection(cap)
    #print(len(img_list),len(record_right),len(record_left),len(right_lost),len(left_lost),len(img_lost))

    fist_closing_frame, action_time_list = find_peak(record_right,record_left)
    print("right hand action count:",len(fist_closing_frame[0]), "regularity:",np.std(np.array(action_time_list[0]))/30)
    print("left hand action count:",len(fist_closing_frame[1]), "regularity:",np.std(np.array(action_time_list[1]))/30)
    print("Write result image to ../demo_result/hand_movement.png")

    #save video
    action_count = [0,0]
    action_time = [0,0]
    right_count = 0
    left_count = 0
    temp_count = 0
    print("Write result video to ../demo_result/hand_movement_result.mp4")
    for i in range(len(img_list)):
        label = ""
        text_loc_x = 0
        for hand in range(2):
            if hand == 0:
                label = "right"
                text_loc_x = 100
                temp_count = right_count
            else:
                label = "left"
                text_loc_x = 1250
                temp_count = left_count
            if i not in right_lost:
                right_count +=1
            if i not in left_lost:
                left_count += 1

            if( (len(fist_closing_frame[hand]) > action_count[hand]) and temp_count == fist_closing_frame[hand][action_count[hand]]):
                action_time[hand] = action_time_list[hand][action_count[hand]]
                action_count[hand] += 1
            cv2.putText(img_list[i],f'action_count({label}):{action_count[hand]}',(text_loc_x, 70), cv2.FONT_HERSHEY_SIMPLEX,1.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img_list[i],f'action_time({label}):{action_time[hand]/fps:.2f}s',(text_loc_x, 120), cv2.FONT_HERSHEY_SIMPLEX,1.5, (0, 0, 0), 3, cv2.LINE_AA)
        out.write(img_list[i])

if __name__ == '__main__':
    demo()
    print("Done")