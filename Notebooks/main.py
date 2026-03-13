import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# add a script argparse param type_ with the following options: "image", "video", "webcam", "interactive_draw", "interactive_circle"


parser = argparse.ArgumentParser(description="Choose the type of OpenCV demo to run.")
parser.add_argument("--type", type=str, choices=["image", "video", "webcam", "interactive_draw", "interactive_circle", "watershed", "detect_face", "sparse_optical_flow", "dense_optical_flow2", "meanshift"], default="interactive_circle", help="Type of demo to run")
args = parser.parse_args()

type_ = args.type

# Read and display an image
if type_ == "image":
    image_path = os.path.join(os.getcwd(), 'images/sample1.png')
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('Image', img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Show a video
if type_ == "video":
    video_path = os.path.join(os.getcwd(), 'images/bird.mp4')
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        cv2.imshow('Video', frame_rgb)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Use the webcam
if type_ == "webcam":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if type_ == "interactive_draw":
    import cv2
    import numpy as np

    drawing = False
    ix, iy = -1, -1


    def draw_rectangle(event, x, y, flags, param):
        
        global ix, iy, drawing
        
        if event == cv2.EVENT_LBUTTONDOWN:
            
            drawing = True
            ix,iy = x,y
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                cv2.rectangle(img, (ix, iy), (x,y), (0,255,0), -1)
                
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(img, (ix, iy), (x,y), (0,255,0), -1)
                

    img = np.zeros((512, 512, 3))

    cv2.namedWindow(winname='my_drawing')

    cv2.setMouseCallback('my_drawing', draw_rectangle)

    while True:
        
        cv2.imshow('my_drawing', img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()

if type_ == "interactive_circle":
    import cv2
    import numpy as np


    def draw_circle(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img, (x,y), 100, (0,255,0), -1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            cv2.circle(img, (x,y), 100, (255,0,0), -1)

    cv2.namedWindow(winname='my_drawing')

    cv2.setMouseCallback('my_drawing', draw_circle)


    img = np.zeros((512,512,3))

    while True:
        cv2.imshow('my_drawing', img)
        
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()

# Watershed algorithm
if type_ == "watershed":
    def create_rgb(i):
        return tuple(np.array(cm.tab10(i)[:3]) * 255)

    colors = []
    for i in range(10):
        colors.append(create_rgb(i))

    road = cv2.imread('data/road_image.jpg')
    road_copy = np.copy(road)
    marker_image = np.zeros(road.shape[:2], dtype=np.int32)
    segments = np.zeros(road.shape, dtype=np.uint8)

    # Color Choice
    n_markers = 9 # 0-9
    current_marker = 1
    # Markers updated by watershed
    marks_updated = False

    # CALLBACK FUNCTION
    def mouse_callback(event, x,y, flags, param):
        global marks_updated
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Markers passed to the watershed call
            cv2.circle(marker_image, (x,y), 10, (current_marker), -1)
            
            # What the user sees on the road image
            cv2.circle(road_copy, (x,y), 10, colors[current_marker], -1)
            
            marks_updated = True

    # WHILE TRUE
    cv2.namedWindow('Road Image')
    cv2.setMouseCallback('Road Image', mouse_callback)

    while True:
        cv2.imshow('Watershed Segments', segments)
        cv2.imshow('Road Image', road_copy)
        
        # CLOSE ALL WINDOWS
        k = cv2.waitKey(1)
        
        if k == ord('q'):
            break
            
        # CLEARING ALL THE COLORS
        elif k == ord('c'):
            road_copy = road.copy()
            marker_image = np.zeros(road.shape[:2], dtype=np.int32)
            segments = np.zeros(road.shape, dtype=np.uint8)
        
        # UPDATE COLOR CHOICE
        # chr turns key press to actual integer
        elif k > 0 and chr(k).isdigit():
            current_marker = int(chr(k))
        
        # UPDATE THE MARKINGS
        if marks_updated:
            marker_image_copy = marker_image.copy()
            cv2.watershed(road, marker_image_copy)
            
            segments = np.zeros(road.shape, dtype=np.uint8)
            
            for color_ind in range(n_markers):
                segments[marker_image_copy==(color_ind)] = colors[color_ind]
        
    cv2.destroyAllWindows()

if type_ == "detect_face":
    face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')


    def detect_face(img):
        face_img = img.copy()
        
        face_rects = face_cascade.detectMultiScale(face_img)
        
        for (x,y,w,h) in face_rects:
            cv2.rectangle(face_img, (x,y), (x+w, y+h), (255,255,255), 10)
            
        return face_img
    

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read(0)
        
        frame = detect_face(frame)
        
        cv2.imshow('Video Face Detect', frame)
        
        k = cv2.waitKey(1)

        if k == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if type_ == "sparse_optical_flow":
    corner_track_params = dict(maxCorners=25, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(200,200), maxLevel=2,  criteria =(cv2.TERM_CRITERIA_EPS | cv2.TermCriteria_COUNT, 10, 0.03))

    cap = cv2.VideoCapture(0)

    ret, prev_frame = cap.read()

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # PTS TO TRACK
    prevPts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **corner_track_params)

    mask = np.zeros_like(prev_frame)

    while True:
        ret, frame = cap.read()
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        nextPts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prevPts, None, **lk_params)
        
        good_new = nextPts[status==1]
        good_prev = prevPts[status==1]
        
        for i, (new,prev) in enumerate(zip(good_new, good_prev)):
            x_new, y_new = new.ravel()
            x_prev, y_prev = prev.ravel()
            
            mask = cv2.line(mask, (x_new, y_new), (x_prev, y_prev), (0,255,0), 3)
            
            frame = cv2.circle(frame, (x_new, y_new), 8, (0,0,255), -1)
            
        img = cv2.add(frame, mask)
        cv2.imshow('tracking', img)
        
        k = cv2.waitKey(30) & 0xFF
        if k == ord('q'):
            break
            
        prev_gray = frame_gray.copy()
        prevPts = good_new.reshape(-1,1,2)
        
    cap.release()
    cv2.destroyAllWindows()


if type_ == "dense_optical_flow2":
    cap = cv2.VideoCapture(0)

    ret, frame1 = cap.read()

    prevImg = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    hsv_mask = np.zeros_like(frame1)
    hsv_mask[:,:,1] = 255

    while True:
        ret, frame2 = cap.read()
        
        nextImg = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(prevImg, nextImg, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:,:,1], angleInDegrees=True)
        
        hsv_mask[:,:,0] = ang/2
        
        hsv_mask[:,:,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        bgr = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
        
        cv2.imshow('frame', bgr)
        
        k = cv2.waitKey(10) & 0xFF
        if k== ord('q'):
            break
            
        prevImg = nextImg
        
    cap.release()
    cv2.destroyAllWindows()

if type_ == "meanshift":

    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()

    while True:
        
        ret, frame = cap.read()
        if not ret:
            break
        face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
        face_rects = face_cascade.detectMultiScale(frame)

        print(face_rects)

        (face_x, face_y, w, h) = tuple(face_rects[0])

        track_window = (face_x, face_y, w, h)

        roi = frame[face_y:face_y+h, face_x:face_x+w]

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0,180])

        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        
        if ret == True:
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)
            
            ##############################################################
            # MEANSHIFT
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)
            
            x,y,w,h = track_window
            img2 = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 5)
            
            # CAMSHIFT
            # ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            
            # pts = cv2.boxPoints(ret)
            # pts = np.int32(pts)
            # img2 = cv2.polylines(frame, [pts], True, (0,0,255), 5)
            ##############################################################
            
            cv2.imshow('img', img2)
            
            k=cv2.waitKey(1) & 0xFF
            if k==ord('q'):
                break
        else:
            break
            
    cap.release()
    cv2.destroyAllWindows()