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

def post_detection(cap):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    pose = mp_pose.Pose(static_image_mode=True,min_detection_confidence=0.4)

    right_shoulder = []
    left_shoulder = []
    frame_count = 0
    lost_frame = []
    img_list = []
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:

            # Convert the BGR image to RGB and process it with MediaPipe Pose.
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            annotated_image = frame.copy()

            if results.pose_landmarks != None:
                mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                right_shoulder.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
                left_shoulder.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y)

            else:
                lost_frame.append(frame_count)

            img_list.append(annotated_image)
            frame_count += 1

        else:
            break

    cap.release()
    return img_list,right_shoulder, left_shoulder, lost_frame

def find_peak(shoulder):
    copy = savgol_filter(shoulder,31,3)
    #copy = right_shoulder.copy()


    mid = (min(copy) + max(copy))/2
    pro = (max(copy) - mid)

    peaks, _ = signal.find_peaks(copy,height=mid,prominence=pro)
    valley, _ = signal.find_peaks(np.array(copy)*-1,height=mid*-1)

    start_frame = peaks[0]
    end_frame = valley[0]

    plt.title("right_shoulder")
    plt.xlabel("frame")
    plt.ylabel("pixel coordinate")

    plt.plot(copy)
    plt.plot(peaks,np.array(copy)[peaks],"x")
    plt.plot(valley,np.array(copy)[valley],"o")
    plt.savefig("./demo_result/arise_from_chair.png")

    return start_frame, end_frame

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
    output_video = './demo_result/arise_from_chair.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, RES)

    img_list,right_shoulder, left_shoulder, lost_frame = post_detection(cap)

    start_frame, end_frame = find_peak(right_shoulder)
    print("Write result image to ./demo_result/arise_from_chair.png")

    #write result to video (arise from chair)
    print("Write result video to ./demo_result/arise_from_chair.mp4")
    has_result = 0
    time_count = 0
    state = "sit"
    for i in range(len(img_list)):
        if i not in lost_frame:
            if has_result == start_frame:
                state = "arise"
            elif has_result == end_frame:
                state = "stand"
            has_result += 1
        if state == "arise":
            time_count+=1

        cv2.putText(img_list[i],f'state:{state}',(10, 70), cv2.FONT_HERSHEY_SIMPLEX,1.5, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img_list[i],f'action_time:{time_count/fps:.2f}s',(10, 120), cv2.FONT_HERSHEY_SIMPLEX,1.5, (255, 0, 0), 3, cv2.LINE_AA)
        out.write(img_list[i])

    out.release()

if __name__ == '__main__':
    demo()
    print("Done")