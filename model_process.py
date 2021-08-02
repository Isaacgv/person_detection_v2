#from utils.utils_model import attempt_load, non_max_suppression, save_one_box, letterbox, scale_coords
import imutils
from imutils.video import FPS
from track import TrackableObject, CentroidTracker  
import numpy as np
#import torch
import cv2
import dlib
import sys
import jetson.inference
import jetson.utils


def load_model():
    print("LOAD MODEL")
    #device = torch.device('cpu')
    #weights = ['models/model/model_detect.pt']
    #model = attempt_load(weights, map_location=device)
    img_size = 640
    #half = False #device.type != 'cpu'
    #if device.type != 'cpu':
    #    model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))

    #stride = int(model.stride.max())
    model = jetson.inference.detectNet("ssd-mobilenet-v2", sys.argv, 0.3)
    return model, img_size #model, img_size, half, stride, device

#model, img_size, half, stride, device = load_model()
model, img_size = load_model()

frame_show = None
placeholder_up = 0
placeholder_down = 0

def detect_person(img_t):
    
    # Padded resize
    #img = letterbox(img_t, img_size, stride=stride)[0]

    # Convert
    #img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    #img = np.ascontiguousarray(img)

    #img = torch.from_numpy(img).to(device)
    #img = img.half() if half else img.float()  # uint8 to fp16/32
    #img /= 255.0  # 0 - 255 to 0.0 - 1.0
    #img = img.unsqueeze(0)

    #pred = model(img, augment=False)[0]
   
    #pred = non_max_suppression(pred, 0.25, 0.45, classes=0, agnostic=True)

    #images_detected = []
    
    #for i, det in enumerate(pred):  # detections per image
    
    #    im0 = img_t.copy()

    #    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    #    if len(det):
            # Rescale boxes from img_size tso im0 size
    #        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    #        return det
    img_t = img_t.cv2.resize(img_t, (img_size, img_size))
    img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)
    img_t = jetson.utils.cudaFromNumpy(img_t)
    detections = model.Detect(img_t, overlay="box,labels,conf")
    print(detections)
    for detec in detections:
        if detec.ClassID == 1:
            startX, startY = detec.Left, detec.Top
            endX, endY = detec.Rights, detec.Bottom
            return (startX, startY, endX, endY), detec.Confidence, detec.ClassID

        

def show_lines(frame, p0_up, p1_up, p0_down, p1_down):
    cv2.line(frame, p0_up, p1_up, (255, 0, 0), 2)
    cv2.line(frame, p0_down, p1_down, (0, 0, 255), 2)
    return frame

def process_detect(vs, p0_up, p1_up, p0_down, p1_down, y_line, path_output=None):
    totalDown = 0
    totalUp = 0
    totalFrames = 0
    trackers = []
    trackableObjects = {}
    writer = None
    W = None
    H = None

    ct = CentroidTracker(maxDisappeared=30, maxDistance=100)
    skip_frames  = 30
    print("START Process")
    fps = FPS().start()

    while True:
        # grab the next frame and handle if we are reading from either
        # VideoCapture or VideoStream
        _, frame = vs.read()
        #frame = frame[1] if args.get("input", False) else frame
        # if we are viewing a video and we did not grab a frame then we
        # have reached the end of the video
        if frame is None:
            break
        # resize the frame to have a maximum width of 500 pixels (the
        # less data we have, the faster we can process it), then convert
        # the frame from BGR to RGB for dlib
        frame = imutils.resize(frame, width=640)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # if the frame dimensions are empty, set them
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        # if we are supposed to be writing a video to disk, initialize
        # the writer
        if path_output is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(path_output, fourcc, 30,
                (W, H), True)
            
        # initialize the current status along with our list of bounding
        # box rectangles returned by either (1) our object detector or
        # (2) the correlation trackers
        status = "Waiting"
        rects = []
        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        if totalFrames % skip_frames == 0:
            # set the status and initialize our new set of object trackers
            status = "Detecting"
            trackers = []
            # convert the frame to a blob and pass the blob through the
            # network and obtain the detections

            
    #		blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
    #		net.setInput(blob)
    #		detections = net.forward()
            
    # loop over the detections
            detections = detect_person(frame)
            #print(detections)   
            if detections != None:       
                for *xyxy, conf, cls in reversed(detections):
                    # extract the confidence (i.e., probability) associated
                    # with the prediction
                    # filter out weak detections by requiring a minimum
                    # confidence
                        
                    # compute the (x, y)-coordinates of the bounding box
                    # for the object
                    (startX, startY, endX, endY) = xyxy
                    # construct a dlib rectangle object from the bounding
                    # box coordinates and then start the dlib correlation
                    # tracker
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)
                    # add the tracker to our list of trackers so we can
                    # utilize it during skip frames
                    trackers.append(tracker)
                    rects.append((startX, startY, endX, endY))
                    
        # otherwise, we should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing throughput
        else:
            # loop over the trackers
            for tracker in trackers:
                # set the status of our system to be 'tracking' rather
                # than 'waiting' or 'detecting'
                status = "Tracking"
                # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()
                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                # add the bounding box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY))

        # draw a horizontal line in the center of the frame -- once an
        # object crosses this line we will determine whether they were
        # moving 'up' or 'down'
     
        cv2.line(frame, p0_up, p1_up, (255, 0, 0), 2)
        cv2.line(frame, p0_down, p1_down, (0, 0, 255), 2)
        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects = ct.update(rects)
        
        
        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = trackableObjects.get(objectID, None)
            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)
            # otherwise, there is a trackable object so we can utilize it
            # to determine direction
            else:
                # the difference between the y-coordinate of the *current*
                # centroid and the mean of *previous* centroids will tell
                # us in which direction the object is moving (negative for
                # 'up' and positive for 'down')
            
                y = [c[1] for c in to.centroids]
                x = [c[0] for c in to.centroids] 
                direction_y = centroid[1] - np.mean(y)
                direction_x = centroid[0] - np.mean(x)
                direction = direction_x
                if y_line:
                    direction = direction_y
                to.centroids.append(centroid)
                # check to see if the object has been counted or not
                d_centroids = np.sqrt(direction_y**2+direction_x**2) 
                if not to.counted and (d_centroids > 2 or d_centroids < -2):
                    # if the direction is negative (indicating the object
                    # is moving up) AND the centroid is above the center
                    # line, count the object
                    distance_up = np.cross(p1_up-p0_up, centroid-p0_up)/np.linalg.norm(p1_up-p0_up)
                    distance_down = np.cross(p1_down-p0_down,centroid-p0_down)/np.linalg.norm(p1_down-p0_down)
                    if (direction < 0 )and distance_up < 0:
                        totalUp += 1
                        to.counted = True
                    # if the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object
                    elif (direction > 0 ) and distance_down > 0:
                        totalDown += 1
                        to.counted = True

            # store the trackable object in our dictionary
            trackableObjects[objectID] = to
            
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)        
        
        fps.update()
        fps.stop()
        info = [
		("Up", totalUp),
		("Down", totalDown),
		("FPS", int(fps.fps())),
        ]
        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # check to see if we should write the frame to disk

        if writer is not None:
            writer.write(frame)
        # show the output frame
         

        placeholder_up = "Up : "+ str(totalUp)
        placeholder_down = "Down : "+ str(totalDown)
   
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        # increment the total number of frames processed thus far and
        # then update the FPS counter
        totalFrames += 1
        #print(totalFrames)

        # Display each frame
        cv2.imshow("video", frame)

        # Quit when 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break
    
    # check to see if we need to release the video writer pointer
    if writer is not None:
        writer.release()

    # otherwise, release the video file pointer

    print(placeholder_up)
    print(placeholder_down)
    #print("Total Frames: ".format(totalFrames))
    #print("FPS: ".format(fps.fps()))
    #print("Time: ".format(fps.elapsed()))

    vs.release()
    # close any open windows
    cv2.destroyAllWindows()

