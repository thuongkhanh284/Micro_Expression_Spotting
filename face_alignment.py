import numpy as np
import cv2
import dlib
from skimage import transform
from skimage import io
from skimage import transform
from skimage import color
from skimage import img_as_ubyte
import os

TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

TEMPLATE_5 = np.float32([])
    
class face_alignment():  
    def __init__(self):
        self.INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
        self.OUTER_EYES_AND_NOSE = [36, 45, 57]
        TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
        self.MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)
        
    def align(self,image_rgb,shape,face_size=225, scale=1.0):
        assert image_rgb is not None
        assert shape is not None
        npLandmarks= self.shape_to_np(shape)
        npLandmarks= self.extra_landmarks(npLandmarks)
        convexhull = cv2.convexHull(npLandmarks)
        mask=np.zeros((image_rgb.shape[0],image_rgb.shape[1]),dtype=np.int8)
        cv2.fillConvexPoly(mask, convexhull, 255)    
        image_rgb = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)  
        npLandmarks = np.float32(npLandmarks)
        npLandmarkIndices = np.array(self.INNER_EYES_AND_BOTTOM_LIP)
        H = cv2.getAffineTransform(npLandmarks[npLandmarkIndices],face_size * TEMPLATE[npLandmarkIndices] * scale + face_size * (1 - scale) / 2)
        face_rgb = cv2.warpAffine(image_rgb, H, (face_size, face_size))
        return face_rgb
        
        
    def face_registration(self, image_rgb,shape,face_size=225, scale=1.0):
        standard_model = TEMPLATE * 500
        if (len(image_rgb.shape) > 2):
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        npLandmarks= self.shape_to_np(shape)
        #npLandmarks= self.extra_landmarks(npLandmarks)
        t = transform.PolynomialTransform()
        t.estimate(standard_model,npLandmarks,3)
        img_warped = transform.warp(image_rgb, t, order=2, mode='constant',cval=float('nan'))
        cropped_registered_face = self.crop_face(img_warped,standard_model)
        
        return img_warped
    
    def face_align_by_5(self, image_rgb,shape,face_size=225, scale=1.0):
        if (len(image_rgb.shape) > 2):
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
        standard_model = TEMPLATE * 500
        TEMPLATE_5 = TEMPLATE[np.array([  39 , 42 , 57 ])]
        npLandmarks= self.shape_to_np(shape)
        npLandmarks = npLandmarks[np.array([ 39 , 42 , 57 ])]
        npLandmarks = np.float32(npLandmarks)
        
        # check the image color or ggrey color image
        H = cv2.getAffineTransform(npLandmarks,face_size * TEMPLATE_5* scale )
        face_rgb = cv2.warpAffine(image_rgb, H, ( face_size , face_size))
        norm_landmark = face_size * TEMPLATE_5* scale
        return face_rgb , norm_landmark

    def crop_face(self , rawImage, landMarks, \
             leftEyeInds = [36, 37, 38, 39, 40, 41], \
             rightEyeInds = [42, 43, 44, 45, 46, 47], \
             do_normalization = False):
        
        
        # should be gray image
        cv_image = img_as_ubyte(rawImage)
        if len(rawImage.shape) == 3 and rawImage.shape[2] == 3:
            rawImage = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # extract the left eye coordinate
        leftEye = landMarks[leftEyeInds, :].sum(0).astype('float')/len(leftEyeInds)
        # extract the right eye coordinate
        rightEye = landMarks[rightEyeInds, :].sum(0).astype('float')/len(rightEyeInds)
        
        # distance between two eyes
        distBetweenEyes = np.sqrt(sum((leftEye - rightEye)**2))
        
        x1 = leftEye[0]
        y1 = leftEye[1]
        x2 = rightEye[0]
        y2 = rightEye[1]

        sina= (y1-y2)/distBetweenEyes
        
        cosa = (x2-x1)/distBetweenEyes
        
        lefttopy = y1 + distBetweenEyes * 0.4 * sina -distBetweenEyes * 0.6 * cosa
        lefttopx = x1 - distBetweenEyes * 0.4 * cosa - distBetweenEyes * 0.6 * sina

        faceHeight = int(round(distBetweenEyes * 2.2))
        faceWidth = int(round(distBetweenEyes * 1.8))
        
        norm_face = np.zeros((faceHeight, faceWidth))

        [wi, hi] = rawImage.shape

        for h in range(0, faceHeight):
            starty = lefttopy + h * cosa
            startx = lefttopx + h * sina
            
            for w in range(0, faceWidth):
                if np.uint16(starty - w * sina) > wi:
                    norm_face[h,w] = rawImage[np.uint16(wi), np.uint16(startx + w * cosa)]
                    
                elif np.uint16(startx + w * cosa) > hi:
                    norm_face[h,w] = rawImage[np.uint16(starty - w * sina), np.uint16(hi)]
                    
                else:
                    norm_face[h,w] = rawImage[np.uint16(starty - w * sina), np.uint16(startx + w * cosa)]

        
        if do_normalization == 1:
            norm_face = transform.resize(norm_face, (128,128))
        
        return norm_face
    
    def shape_to_np(self, shape):
        coords = np.zeros((68, 2), dtype='int')
        for i in range(0, 68):
            coords[i,:] = [shape.part(i).x, shape.part(i).y]
        return coords
    
    def extra_landmarks(self, shape):
        shape= np.vstack((shape,np.int32(shape[17:27,:]-[0,shape[30,1]-shape[27,1]])))
        return shape 
        
class video_face_alignment():
    def __init__(self):
        self.INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
        self.OUTER_EYES_AND_NOSE = [36, 45, 57]
        TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
        self.MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)


