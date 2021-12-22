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
parser.add_argument("-l","--leg",type=str,help="the right or left lag",required=True)
args = parser.parse_args()

def post_detection(cap):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    pose = mp_pose.Pose(static_image_mode=True,min_detection_confidence=0.4)

    foot_index = []
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

                if(args.leg == "right"):
                    foot_index.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y)
                elif (args.leg == "left"):
                    foot_index.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y)

            else:
                lost_frame.append(frame_count)

            img_list.append(annotated_image)
            frame_count += 1

        else:
            break

    cap.release()
    return img_list,foot_index, lost_frame

def find_peak(foot_index):
    action_time_list = []

    smooth = foot_index.copy()
    middle = (max(smooth)+min(smooth))/2
    pro = (max(smooth) - middle) * 0.4

    peaks, _ = signal.find_peaks(smooth,height=middle,prominence=pro)

    plt.plot(smooth)
    plt.plot(peaks,np.array(smooth)[peaks],"x")

    #calculcate stomp action time
    action_time_list = []
    action_time_list.append(peaks[0])
    for time in np.diff(peaks):
        action_time_list.append(time)

    plt.savefig('./demo_result/' + args.leg +'_leg_agility.png')
    return peaks, action_time_list

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
    output_video = './demo_result/' + args.leg +'_leg_agility.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, RES)

    img_list,foot_index, lost_frame = post_detection(cap)

    stmop_list, action_time_list = find_peak(foot_index)
    print(f'{args.leg} leg action count:{len(stmop_list)}, frequency:{len(stmop_list)/(len(img_list)/fps):.4f}, regularity:{np.std(np.array(action_time_list))/fps:.4f}')
    print("Write result image to ./demo_result/" + args.leg +  "leg_agility.png")

    #write result to video (leg agility)
    print("Write result video to ./demo_result/" + args.leg +  "leg_agility.mp4")
    action_count = 0
    action_time = 0
    has_result = 0
    for i in range(len(img_list)):
        if i not in lost_frame:
            if( len(stmop_list) > action_count and has_result == stmop_list[action_count]):
                action_time = action_time_list[action_count]
                action_count += 1
            has_result += 1

        cv2.putText(img_list[i],f'action_count:{action_count}',(10, 70), cv2.FONT_HERSHEY_SIMPLEX,1.5, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img_list[i],f'action_time:{action_time/fps:.2f}s',(10, 120), cv2.FONT_HERSHEY_SIMPLEX,1.5, (255, 0, 0), 3, cv2.LINE_AA)
        out.write(img_list[i])

    out.release()

if __name__ == '__main__':
    demo()
    print("Done")