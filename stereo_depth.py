import numpy as np
import cv2
import argparse
import os
import sys
from calibration_store import load_stereo_coefficients
from PIL import Image 
import PIL
import matplotlib.pyplot as plt
import time 
from open3d import *
import open3d as o3d 



#====================================
#Function to create point cloud file
def create_output(vertices, filename):
	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num=len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')

def depth_map(dispMap,orignal_pic):
    print("Calculating depth....")
    depth = np.zeros(dispMap.shape)
    coordinates=[]
    h,w = dispMap.shape
    h1,w1,_ = orignal_pic.shape
    print( h,w ,"-----", h1,w1)

    for r in range(0,h):
        for c in range(0,w):
            disparity= dispMap[r,c]
            Yoffset=((h-r)*2)-Y
            Xoffset=((w-c)*2)-X_A
            depth[r,c] =  (CAMERA_DISTANCE * FOCAL_LENGTH) / (dispMap[r,c])
            # This will contain x,y,z coordinates with R,G,B values for the pixel
            ZZ=(CAMERA_DISTANCE*FOCAL_LENGTH)/(disparity+100)
            YY=(ZZ/FOCAL_LENGTH)*Yoffset
            XX=(ZZ/FOCAL_LENGTH)*Xoffset
            coordinates+=[[XX,YY,ZZ,orignal_pic[r][c][2],orignal_pic[r][c][1],orignal_pic[r][c][0]]]
    depthmap = plt.imshow(depth,cmap='jet_r')
    return coordinates


def generate_window(row, col, image, blockSize):
    window = (image[row:row + blockSize, col:col + blockSize])
    return window


def disparitymap(imgL,imgR,dispMap=[]):
    # Size of the search window 
    blockSize = 5
    h, w, _ = imgL.shape
    dispMap = np.zeros((h, w))
    # maximum disparity to search for (Tuned by experimenting)
    max_disp = int(w//3)
    # Initializing disparity value 
    dispVal = 0
    tic=time.time()
    for row in range(0, h - blockSize + 1, blockSize):
        for col in range(0, w - blockSize + 1, blockSize):
            winR = generate_window(row, col, imgR, blockSize)
            sad = 9999
            dispVal = 0
            for colL in range(col + blockSize, min(w - blockSize, col + max_disp)):
                winL = generate_window(row, colL, imgL, blockSize)
                tempSad = int(abs(winR- winL).sum())
                if tempSad < sad:
                    sad = tempSad
                    dispVal = abs(colL - col)
            for i in range(row, row + blockSize):
                for j in range(col, col + blockSize):
                    dispMap[i, j] = dispVal

  
        # Updating progress 
        if (row % 50 == 0):
            print('Row number {} Percent complete {} %'.format(row,row*100/h))
    toc = time.time()
    print('elapsed time... {} mins'.format((toc - tic)/60 ))
    #printing the disparity amap
    print("Disparity map....\n")
    plt.title('Disparity Map')
    plt.ylabel('Height {}'.format(dispMap.shape[0]))
    plt.xlabel('Width {}'.format(dispMap.shape[1]))
    # plt.imshow(dispMap,cmap='gray')
    # plt.show()

    return dispMap


def dispar_map(imgL, imgR):
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( rigth to left disparity ) """
    # SGBM Parameters -----------------
    window_size = 3 # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=5*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.3
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    return filteredImg

def show3d(dir):
    pcd = o3d.io.read_point_cloud(dir+'pointcloud.ply')
    o3d.visualization.draw_geometries([pcd])
    return

if __name__ == '__main__':
    # Args handling -> check help parameters to understand
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--calibration_file', type=str, required=True, help='Path to the stereo calibration file')
    parser.add_argument('--left_source', type=str, required=True, help='Left image')
    parser.add_argument('--right_source', type=str, required=True, help='Right image')
    parser.add_argument('--pointcloud_dir', type=str, required=True, help=' directory path to save pointcloud')

    args = parser.parse_args()



    leftFrame = cv2.imread(args.left_source)
    rightFrame = cv2.imread(args.right_source)

    #====================================
    # Get cams params from file "stereo.yml"
    K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q = load_stereo_coefficients(args.calibration_file) 
    height, width, channels = rightFrame.shape  # We will use the shape for remap

    #======================================
    # Factor for downscaling of test images
    SCALE = 1
    #=====================================
    # CAMERA PARAMS
    # The focal length of the two cameras, taken from stereo calibration file
    FOCAL_LENGTH = Q[2][3]     #FOCAL_L 

    # The distance between the two cameras, taken from stereo calibration file
    X_A=  Q[0][3]                #C_X CAMERA L 
    X_B=  K1[0][2]               #C_X CAMERA R 
    Y=    Q[1][3]                 #C_Y CAMERA L 
    DOFFS = X_B-X_A
    CAMERA_DISTANCE = 300   
    #=====================================
    # Function declarations
    ndisp=64
    vmin=0
    #====================================
    # Undistortion and Rectification part!
    scale_percent = 100

    widthR = int(rightFrame.shape[1] * scale_percent / 100)
    heightR = int(rightFrame.shape[0] * scale_percent / 100)
    dsizeR = (widthR, heightR)
    ResizedrightFrame = cv2.resize(rightFrame, dsizeR)
 

    widthL = int(leftFrame.shape[1] * scale_percent / 100)
    heightL = int(leftFrame.shape[0] * scale_percent / 100)
    dsizeL = (widthL, heightL)
    ResizedleftFrame = cv2.resize(leftFrame, dsizeL)
   
    #==============================================
    #rectified images
    leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (int(width*1.), int(height*1.)), cv2.CV_32FC1)
    left_rectified = cv2.remap(ResizedleftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (int(width*1.), int(height*1.)), cv2.CV_32FC1)
    right_rectified = cv2.remap(ResizedrightFrame, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    # We need grayscale for disparity map.
    gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

    disparity_map =  dispar_map(gray_right,gray_left)  # Get the disparity map builded by SGBM
    # disparity_MAP2=  disparitymap(leftFrame,rightFrame)  # Get the disparity map builded by SAD Block Matching

    path = args.pointcloud_dir
    cv2.imwrite(os.path.join(path , 'disparity_image.jpg'), disparity_map)
    cv2.waitKey(0)

    coordinates= depth_map(disparity_map,rightFrame)
    print('\n Creating the output file... \n')
    create_output(coordinates,path+'pointcloud.ply')
    print('\n Done \n')
    show3d(path)