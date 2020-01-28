import cv2
import numpy as np
import dlib
from face_alignment import face_alignment
import os
import util
import feature
import math
import time

# this source code is used to detect micro-expression by combining LBP-Spotting and CNN method
    
    
def divide_image_to_block(gray_img , norm_landmark):

    norm_landmark = np.int8(norm_landmark)
    size_rect = 40
    st_i = norm_landmark[0][0] - size_rect * 2
    en_i = norm_landmark[1][1] + size_rect * 2

    pos_x = []
    for i in range(5):
        pos_x.append([st_i, st_i + 41 ])
        st_i = st_i + 2
    pos_y = []
    st_i = norm_landmark[0][1] - size_rect * 2
    for i in range(3):
        pos_y.append([st_i, st_i + 41 ])
        st_i = st_i + 1
    # pos_x = [[1,26],[27,52],[53,78],[79,104],[105,130],[131,156]]
    # pos_y = [[1,26],[27,52],[53,78],[79,104],[105,130],[131,156]]
    list_img_block = []
    for xa in pos_x:
        for ya in pos_y:
            row_st = xa[0]
            row_en = xa[1]

            col_st = ya[0]
            col_en = ya[1]
            
            
            img_block = gray_img[col_st:col_en,row_st: row_en]

            list_img_block.append(img_block)

    pos_x = []
    pos_y = []
    st_i = norm_landmark[2][0] - size_rect * 2 - 20 
    for i in range(5):
        pos_x.append([st_i, st_i + 41 ])
        st_i = st_i + 1
    pos_y = []
    st_i = norm_landmark[2][1] - size_rect  - 10
    for i in range(3):
        pos_y.append([st_i, st_i + 41 ])
        st_i = st_i + 1
    return list_img_block


def calculate_distance_block( curr_img, hframe, tframe ,norm_landmark):

    blocks_curr = divide_image_to_block(curr_img  , norm_landmark)
    blocks_head = divide_image_to_block(hframe , norm_landmark)
    blocks_tail = divide_image_to_block(tframe , norm_landmark)
    num_block = len(blocks_curr)
    dist_list = []
    
    for iblock in range(0,num_block):
        blck_curr = blocks_curr[iblock]
        blck_hfrm = blocks_head[iblock]
        blck_tfrm = blocks_tail[iblock]
        lbp_feat_1 = feature.extract_LBP_feature(blck_curr)
    
        lbp_feat_2 = feature.extract_LBP_feature(blck_hfrm)
        lbp_feat_3 = feature.extract_LBP_feature(blck_tfrm)

        avg_feat = np.add(lbp_feat_2,lbp_feat_3)/2
        
        dist = ChiSquare_dist(lbp_feat_1,avg_feat)
            
        if (math.isnan(dist)):
            dist = 10.0
        dist_list.append(dist)
    np_dist = np.array(dist_list)
    dist_list.sort()
    idx_sorted = np.argsort(np_dist)
    sum_dist = 0.0
    for i in range(num_block-12,num_block):
        sum_dist = sum_dist + dist_list[i]
    return sum_dist, idx_sorted


def ChiSquare_dist( np1, np2):
    np_shape = np1.shape[0]
    dist = 0.0
    for i in range(0,np_shape):
        if (np1[i] + np2[i] == 0):
            xs = 0.0
        else:
            xs = (np1[i] - np2[i])*(np1[i] - np2[i]) / (np1[i] + np2[i])

        dist = dist + xs

    return float (dist[0])


def calculate_curr_thr ( Farr  , num_frame , window_len):

    frames = num_frame
    Carr = np.zeros(frames)
    L = int(window_len/2)
    for i in range(L,frames-L):
        Carr[i] = Farr[i] - 0.5*(Farr[i-L] + Farr[i+L-1])
        if (Carr[i] < 0):
            Carr[i] = 0

    Cmean = np.mean(Carr)
    Cmax = np.max(Carr)
    epsilon = 0.6
    Thr = Cmean + epsilon * (Cmax -  Cmean)
    for i in range(frames-2*L,frames-L):
        if (i > 30):
            if (Carr[i] >= Thr):
                return int(1)

    return int(0)



def detect_micro_LBP(video_file_path ):
    print(' Processing Micro-expression on: ', video_file_path)
    
    detector = dlib.get_frontal_face_detector()
    predictor_1 = dlib.shape_predictor('.//Models//shape_predictor_5_face_landmarks.dat')
    predictor_2 = dlib.shape_predictor('.//Models//shape_predictor_68_face_landmarks.dat')
    
    cap = cv2.VideoCapture(video_file_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps= int(cap.get(cv2.CAP_PROP_FPS))
    print(fps)
    frames= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    


    face_location=None
    window_len = int (fps / 2)
    processing_list = []
    falign = face_alignment()
    Farr = np.zeros(frames)
    micro_array = np.zeros(frames)
    for i in range(frames):
        flag, frame = cap.read()        
        if (flag==True):
            
            start_time = time.time()
            dets = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 0)
            if (len(dets) > 0):
                big_face=0
                for j,d in enumerate(dets):
                    if (d.right()-d.left()>big_face):
                        big_face=d.right()-d.left()
                        face_location=dets[j]
                shape_2 = predictor_2(frame, face_location)
                    

                #shape_1 = predictor_1(frame, face_location)
                #aligned_face = falign.face_registration(frame, shape_2, 225, 1.0)
                aligned_face , norm_landmark = falign.face_align_by_5(frame, shape_2, 255, 1.0)
                processing_list.append(aligned_face)
        
                count =  len(processing_list)
                if (count > window_len):
                    processing_list.pop(0)

                count =  len(processing_list)
                L = int(window_len/2) 
                idx = i
                dist = 0.0
                if (count == window_len):
                    st_idx = idx - window_len + 1
                    apex_idx = idx - int(window_len/2)

                    hframe = processing_list[0]
                    tframe = processing_list[count-1]

                    curr_frame = processing_list[int(window_len/2)]

                    dist , idx_sorted = calculate_distance_block(curr_frame,hframe,tframe , norm_landmark)

                    pos =  idx - int(window_len/2)
                    Farr[pos] = dist
                    res = calculate_curr_thr(Farr, i+1 , window_len)

                    if (res == 1):
                        print(pos)


                pre_shape = shape_2


    return 1

def detect_micro_LBP_Webcam( ):
    print(' Processing Micro-expression by webcam ')
    
    detector = dlib.get_frontal_face_detector()
    
    predictor_2 = dlib.shape_predictor('.//Models//shape_predictor_68_face_landmarks.dat')
    
    cap = cv2.VideoCapture(0)
    window_len = int (30 / 2)
    processing_list = []
    falign = face_alignment()
    frames = 2000
    Farr = np.zeros(frames)
    micro_array = np.zeros(frames)
    i = 0
    face_location=None
    res = 0
    while True:
        
        
        flag, frame = cap.read()        
        if (flag==True):
            
            dets = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 0)
            if (len(dets) > 0):
                big_face=0
                for j,d in enumerate(dets):
                    if (d.right()-d.left()>big_face):
                        big_face=d.right()-d.left()
                        face_location=dets[j]
                shape_2 = predictor_2(frame, face_location)
                    

                #shape_1 = predictor_1(frame, face_location)
                #aligned_face = falign.face_registration(frame, shape_2, 225, 1.0)
                aligned_face , norm_landmark = falign.face_align_by_5(frame, shape_2, 255, 1.0)
                processing_list.append(aligned_face)
        
                count =  len(processing_list)
                if (count > window_len):
                    processing_list.pop(0)

                count =  len(processing_list)
                L = int(window_len/2) 
                idx = i
                i = i + 1
                dist = 0.0
                if (count == window_len):
                    st_idx = idx - window_len + 1
                    apex_idx = idx - int(window_len/2)

                    hframe = processing_list[0]
                    tframe = processing_list[count-1]

                    curr_frame = processing_list[int(window_len/2)]

                    dist , idx_sorted = calculate_distance_block(curr_frame,hframe,tframe , norm_landmark)

                    pos =  idx - int(window_len/2)
                    Farr[pos] = dist
                    res = calculate_curr_thr(Farr, i+1 , window_len)

                    if (res == 1):
                        print(pos)


                pre_shape = shape_2

            shape_np = falign.shape_to_np(shape_2)
            d=shape_np[45,0]-shape_np[36,0] 
            if (res == 1):
                frame = cv2.rectangle(frame, (shape_np[30,0]-d,shape_np[30,1]-d) , (shape_np[30,0]+d,shape_np[30,1]+d) ,(0,255,0), 1) 
            cv2.imshow('Face Micro expression demo ',frame)          
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  

    return 1

def run_real_time_file():
    
    return 1
def run_live_demo():
    return 1

def parser_arguments():

def main():
    video_name = 'CASME2.mp4';
    folder_video = './/dataset//'
    video_path = os.path.join(folder_video,video_name)
    #res = detect_micro_LBP(video_path)
    detect_micro_LBP_Webcam()
    return 0
    
if __name__ == '__main__':
    main()  