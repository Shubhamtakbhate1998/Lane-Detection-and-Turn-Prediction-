import cv2
import matplotlib.pyplot as plt
import numpy as np
size = (960, 540)

#out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize)
out = cv2.VideoWriter('Problem2.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

cap=cv2.VideoCapture("/home/shubham/whiteline.mp4")
r=960
c=540
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def perspective_warp(img,
                     dst_size=(r,c),
                     src=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)]),
                     dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
  
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped
def inv_perspective_warp(img, 
                     dst_size=(r,c),
                     src=np.float32([(0,0), (1, 0), (0,1), (1,1)]),
                     dst=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
 
    dst = dst * np.float32(dst_size)
  
    M = cv2.getPerspectiveTransform(src, dst)
   
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped,M
while(cap.isOpened()):
 
  ret, frame = cap.read()
 
  
  if ret == True:
    blur=cv2.blur(frame, (5,5))
    erode=cv2.erode(frame, (5,5))
    w=perspective_warp(erode,
                     dst_size=(r,c),
                     src=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)]),
                     dst=np.float32([(0,0), (1, 0), (0,1), (1,1)]))
    gray=cv2.cvtColor(w, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
    for contour in contours:
            area = cv2.contourArea(contour)
            if area >10000:
                approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
                cv2.drawContours(w, [approx], 0, (0, 255, 0), thickness=cv2.FILLED)
            else:
                approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
                cv2.drawContours(w, [approx], 0, (0, 0, 255), thickness=cv2.FILLED)
    w2,M=inv_perspective_warp(w, 
                     dst_size=(r,c),
                     src=np.float32([(0,0), (1, 0), (0,1), (1,1)]),
                     dst=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)]))
                
    cv2.imshow("Birdview",w)
    dest_or = cv2.bitwise_or(frame, w2, mask = None)
    #dest_or = cv2.bitwise_and(frame, dest_or, mask = None)
    
  
    for i in range(540):
      for j in range(960):
        if (np.int0(w2[i][j]) == (0, 0, 0)).all():
          w2[i,j]=frame[i,j]

    cv2.imshow("Detected_lane", w2)
    out.write(w2)

    

   
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  
  else: 
    break

cap.release()
cv2.destroyAllWindows()