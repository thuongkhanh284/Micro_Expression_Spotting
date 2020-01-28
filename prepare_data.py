import cv2
import numpy as np
import dlib
from face_alignment import face_alignment
import os

DATABASE_FOLDER = '.\\'

test_video = '.\\dataset\\sub-02-02.mp4'

ROOT_OUT_FOLDER = 'cropped'

def str_idx(x ):

    if (x<10):
        return '000' + str(x)
    else:
        if (x< 100):
            return '00' + str(x)
        else:
            if (x<1000):
                return '0' + str(x)
            else:
                return str(x)

def extract_face_alignment( video_path , output_dir ):
    detector = dlib.get_frontal_face_detector()
    predictor_1 = dlib.shape_predictor('.\\Models\\shape_predictor_5_face_landmarks.dat')
    predictor_2 = dlib.shape_predictor('.\\Models\\shape_predictor_68_face_landmarks.dat')
    
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps=cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frames= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scale=1280/width 
    Write_frames = cv2.VideoWriter('Videos/tmp/frames.avi',fourcc, fps, (int(width*scale),int(height*scale)))   
    falign = face_alignment()
    for i in range(frames):
        flag, frame = cap.read()        
        if (flag==True):
            frame = cv2.resize(frame,(int(width*scale),int(height*scale)))
            
            
            dets = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 0)
            if (len(dets) > 0):
                big_face=0
                for j,d in enumerate(dets):
                     if (d.right()-d.left()>big_face):
                        big_face=d.right()-d.left()
                        face_location=dets[j]
                shape_1 = predictor_1(frame, face_location)
                shape_2 = predictor_2(frame, face_location)
                aligned_face , norm_land = falign.face_align_by_5(frame, shape_2, 255, 1.0)
                filename = 'img' +  str_idx(i) + '.jpg' 
                filename = os.path.join(output_dir,filename)
                cv2.imwrite(filename, aligned_face) 
    return 1

# this function is used to create Face Registration based 

def run_process_folder (in_folder  ):

    list_videos = os.listdir(in_folder)

    for video_name in list_videos:
        video_path = os.path.join(in_folder , video_name)
        pre_video_name = video_name.split('.')[0]
        out_video_path = os.path.join(ROOT_OUT_FOLDER , pre_video_name)
        print(out_video_path)
        if (os.path.isdir(out_video_path) == False ):
            os.mkdir(out_video_path)
        extract_face_alignment(video_path , out_video_path)

def main():
    in_folder = '.\\dataset\\me-cuts\\cuts\\'
    run_process_folder(in_folder)
    #extract_face_alignment(test_video)
    

if __name__ == '__main__':
    main()  