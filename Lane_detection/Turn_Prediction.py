import numpy as np
import cv2
from numpy.linalg import inv
import matplotlib.pyplot as plt
import glob


nwindows = 9
margin=110 
minpix=20
source = np.array([[499, 40], [680, 40], [1078, 250], [231, 257]], dtype="float32") #source for homography
destination = np.array([[50, 0], [250, 0], [250, 500], [0, 500]], dtype="float32") # Desination for homography
ym_per_pix = 3*8/1280 # meters per pixel in y dimension, 8 lines (5 spaces, 3 lines) at 10 ft each = 3m
xm_per_pix = 3.7/720 # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
size = (1280, 720)

#out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize)
out = cv2.VideoWriter('Problem3.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

#Find the homogrphy between source and destination points
def find_homography(src,dst):
   Homgraphy_matrix = cv2.getPerspectiveTransform(src, dst)
   Inverse_homography_matrix = inv(Homgraphy_matrix)
   return Homgraphy_matrix, Inverse_homography_matrix
H_matrix,Hinv=find_homography(source, destination) 
#Function to predict turn and approxmiately determining the center position
def turn_direction(image_center, right_lane_pos, left_lane_pos):
    lane_center = left_lane_pos + (right_lane_pos - left_lane_pos)/2
    
    if (lane_center - image_center < 0):
        return ("Turning left")
    elif (lane_center - image_center < 8):
        return ("straight")
    else:
    	return ("Turning right")
#Function to calibrate
def camera_chessboards(glob_regex='calibration*.jpg'):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    chessboards = [] # array of chessboard images
    
    # Make a list of calibration images
    images = glob.glob(glob_regex)

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners, ret)
            chessboards.append(img)
        
    return objpoints, imgpoints, chessboards
a,b,c=camera_chessboards()    
def camera_calibrate(objpoints, imgpoints, img):
    # Test undistortion on an image
    img_size = img.shape[0:2]

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    
    return ret, mtx, dist, dst
test_img='/home/shubham/Desktop/673/proj2/calibration1.jpg'
img = cv2.imread(test_img)   
ret, mtx, dist, dst = camera_calibrate(a, b, img) 
def undistort_image(img, mtx=mtx, dist=dist):
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst
def turn_direction(image_center, right_lane_pos, left_lane_pos):
    lane_center = left_lane_pos + (right_lane_pos - left_lane_pos)/2
    
    if (lane_center - image_center < 0):
        return ("Turning left")
    elif (lane_center - image_center < 8):
        return ("straight")
    else:
    	return ("Turning right")

def line_detect(img):
        
      
        region_of_interest = img[420:720, 40:1280, :] 
        undist_img = undistort_image(region_of_interest)
        hsl_img = cv2.cvtColor(undist_img, cv2.COLOR_BGR2HLS)
        #Filter out the yellow lane from the cropped image
        lower_mask_yellow = np.array([20, 120, 80], dtype='uint8')
        upper_mask_yellow = np.array([45, 200, 255], dtype='uint8')
        mask_yellow = cv2.inRange(hsl_img, lower_mask_yellow, upper_mask_yellow)
    
        yellow_detect = cv2.bitwise_and(hsl_img, hsl_img, mask=mask_yellow).astype(np.uint8)

        # Filter out white lines
        lower_mask_white = np.array([0, 200, 0], dtype='uint8')
        upper_mask_white = np.array([255, 255, 255], dtype='uint8')
        mask_white = cv2.inRange(hsl_img, lower_mask_white, upper_mask_white)

        white_detect = cv2.bitwise_and(hsl_img, hsl_img, mask=mask_white).astype(np.uint8)

        # Combine both using biwise or
        lanes = cv2.bitwise_or(yellow_detect, white_detect) 
        new_lanes = cv2.cvtColor(lanes, cv2.COLOR_HLS2BGR)
        final = cv2.cvtColor(new_lanes, cv2.COLOR_BGR2GRAY)
        #Blur for noise reduction
        img_blur = cv2.bilateralFilter(final, 9, 120, 100)
        #Canny edge detction for finding edges
        img_edge = cv2.Canny(img_blur, 100, 200)
        new_img = cv2.warpPerspective(img_edge, H_matrix, (300, 600))
        


       
        
        # Find histogram
        histogram = np.sum(new_img, axis=0)
        out_img = np.dstack((new_img,new_img,new_img))*255
        #Determine center of the histogram
        midpoint = np.int64(histogram.shape[0]/2)
    
        #compute left and right pixel
        leftx_ = np.argmax(histogram[:midpoint])
        rightx_ = np.argmax(histogram[midpoint:]) + midpoint
        #print(leftx_base)
        
        left_lane_pos = leftx_
        right_lane_pos = rightx_
        image_center = int(new_img.shape[1]/2)

        # Predict the lane direction as per the curvature
        prediction = turn_direction(image_center, right_lane_pos, left_lane_pos)	
                
        window_height = np.int64(new_img.shape[0]/nwindows)

        nonzero = new_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Update current position for each window
        leftx_p = leftx_
        rightx_p = rightx_
        
        # left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        # Iterate over the windows
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_down = new_img.shape[0] - (window+1)*window_height
            win_y_up = new_img.shape[0] - window*window_height
            win_x_left_down = leftx_p - margin
            win_x_left_up = leftx_p + margin
            win_x_right_down = rightx_p - margin
            win_x_right_up = rightx_p + margin
            
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_down) & (nonzeroy < win_y_up) & (nonzerox >= win_x_left_down) & (nonzerox < win_x_left_up)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_down) & (nonzeroy < win_y_up) & (nonzerox >= win_x_right_down) & (nonzerox < win_x_right_up)).nonzero()[0]
            
            # Append these indices to the list
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If found > minpix pixels, move to next window
            if len(good_left_inds) > minpix:
                leftx_p = np.int64(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_p = np.int64(np.mean(nonzerox[good_right_inds]))
        
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 


        if leftx.size == 0 or rightx.size == 0 or lefty.size == 0 or righty.size == 0:
            return

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        ploty = np.linspace(0, new_img.shape[0]-1, new_img.shape[0] )

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        # Fit a second order polynomial to each
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        
       
        
        # Extract points from fit
        left_line_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right_line_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx,
                                ploty])))])

        image_center = img_edge.shape[0]/2

        
        pts = np.hstack((left_line_pts, right_line_pts))
        pts = np.array(pts, dtype=np.int32)

        color_blend = np.zeros_like(img).astype(np.uint8)
        cv2.fillPoly(color_blend, pts, (0,255, 0))

        
        # Project the image back to the orignal coordinates
        newwarp = cv2.warpPerspective(color_blend, Hinv, (region_of_interest.shape[1], region_of_interest.shape[0]))
        result = cv2.addWeighted(region_of_interest, 1, newwarp, 0.5, 0)
        
        cv2.putText(result, prediction, (200, 50),cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0,255,0),2, cv2.LINE_AA)
        y_eval = np.max(ploty)
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Show the output image
        R= ("Radius")+str(int(left_curverad+right_curverad/2))+("")+("meters")
        cv2.putText(result, str(R), (200, 200),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2, cv2.LINE_AA)
        return result



cap=cv2.VideoCapture("/home/shubham/challenge1.mp4")
while(cap.isOpened()):
    ret, image = cap.read()
  
    if ret == True:
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       
        
        #out.write(result)
        result=line_detect(image)
        out.write(result)
        size = image[0].shape[:2]
        
        cv2.imshow("Result", result)
        
        

        
       
      
       
      
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break
    else:
        break  
   