import cv2
import numpy as np



cap = cv2.VideoCapture("C:\\Users\\38095\\Downloads\\Test task1_video.mp4")
ret, frame = cap.read()
h, w, _ = frame.shape
frameTime = 100

fourcc = cv2.VideoWriter_fourcc(*"XVID")
fps = 1000 / frameTime
writer = cv2.VideoWriter("output.mp4", fourcc, fps, (w, h))
while ret:


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    image=cv2.medianBlur(frame, 5)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([34, 0, 120], dtype="uint8")
    upper = np.array([88, 213, 187], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask,kernel,iterations = 2)
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    colorDict={"Triangle":(255,0,0),"Rectangle":(0,255,0),"Circle":(0,0,255)}
    for cnt in range(len(contours)):

        if len(contours[cnt])<100:
            continue

        epsilon = 0.01 * cv2.arcLength(contours[cnt], True)
        approx = cv2.approxPolyDP(contours[cnt], epsilon, True)


        shape_type="Other"
        corners = len(approx)

        if corners == 3:
            shape_type = "Triangle"
        if corners == 4:
            shape_type = "Rectangle"
        if corners >= 10:
            shape_type = "Circle"
        if shape_type!="Other":
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1
            color = colorDict[shape_type]
            thickness = 2
            cv2.drawContours(frame, contours, cnt, color, 3)
            frame = cv2.putText(frame, shape_type, approx[0][0], font,
                                fontScale, color, thickness, cv2.LINE_AA)


    cv2.imshow('video', image)
    writer.write(frame)
    ret, frame = cap.read()



