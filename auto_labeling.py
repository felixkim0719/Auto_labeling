import cv2 #opencv 임포트
import numpy as np 
import time
capture = cv2.VideoCapture(1)  #카메라 화면 녹화
if not capture.isOpened():
    print("Cannot open camera")
    exit()
w1 = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
h1 = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("원본 동영상 너비(가로) : {}, 높이(세로) : {}".format(w1, h1))
d=0
rx=0
ry=0
w=0
h=0
color=(120,0,0)#원의 색
flie = "C://Users//82102//Desktop//label//classes.txt"
f = open(flie, 'w')
data = "construction\n"
f.write(data)   # data를 파일에 쓰기
data = "intersection\n"
f.write(data)
data = "level crossing\n"
f.write(data)   # data를 파일에 쓰기
data = "parking\n"
f.write(data)   # data를 파일에 쓰기
data = "tunnel"
f.write(data)   # data를 파일에 쓰기
answer = int(input("construction:0 intersection:1 level crossing:2 parking:3 tunnel:4 누르시오"))
while answer == 1:
   # try:
        
        ret, src2 = capture.read() #src2 에 카메라 화면 저장
        img2 = cv2.cvtColor(src2,cv2.COLOR_BGR2GRAY) # 위 변수를 흑백으로 변환

                        #표지판은 원 모양이니 원 모양 찾아냄.특히 param1값을 환경에 맞쳐 바꾸는 걸 추천(높을수록 민감)
        circles = cv2.HoughCircles(img2,cv2.HOUGH_GRADIENT,1,20,param1=115, param2=90,minRadius=1,maxRadius = 500)
                       
        dst = src2.copy()#카메라 화면 복사해서 저장
        src2_2= src2.copy() 
                     
        if circles is not None:#원모양 찾으면
            
            for i in circles[0]:
                d=d+1#flie name
                #사진 경로
                cv2.imwrite('C://Users//82102//Desktop//label//%s (%d).jpg'%(answer,d),dst)
                print(circles[0])
                array_1 = np.array(circles[0])
                col_x = array_1[0, 0]
                col_y = array_1[0, 1]
                col_r = array_1[0, 2]
                #화면 크기에 대한 라벨링 숫자 계산
                x = float(col_x)/w1
                y = float(col_y)/h1
                w = float(col_r)/(w1/2)
                h = float(col_r)/(h1/2)
                print(x," ",y," ",w," ",h)
                #라벨 경로
                flie = "C://Users//82102//Desktop//label/%s (%d).txt"%(answer,d)
                f = open(flie, 'w')
                data=str(answer)
                f.write(data)
                f.write(' ')
                data=str(x)
                f.write(data)
                f.write(' ')
                data=str(y)
                f.write(data)
                f.write(' ')
                data=str(w)
                f.write(data)
                f.write(' ')
                data=str(h)
                f.write(data)
                cv2.imshow("image",img2)
        else:#원이 없으면
            #print("원 없음")  
            continue 
    
        
    #except:
     #   print("error")
while answer == 0 or answer == 4:
    # 프레임을 읽습니다.
    ret, frame = capture.read()
    dst = frame.copy()
    
    # 이미지를 HSV 색상 모델로 변환합니다.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 노란색 색상 범위를 지정합니다.
    lower_yellow = np.array([30, 255, 255])
    upper_yellow = np.array([30, 255, 255])

    # 노란색 영역을 찾습니다.
    kernel = np.ones((3, 3), np.float32) / 9  #(4, 4), /16
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_yellow= cv2.dilate(mask_yellow, kernel, iterations=2)

    # 검은색 색상 범위를 지정합니다.
    lower_black = np.array([160, 100, 100])
    upper_black = np.array([179, 255, 255])

    # 검은색 영역을 찾습니다.
    kernel2 = np.ones((4, 4), np.float32) / 16  #(4, 4), /16
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    mask_black= cv2.dilate(mask_black, kernel2, iterations=2)



    # 노란색과 검은색 영역을 합칩니다.
    mask_combined = cv2.bitwise_or(mask_yellow, mask_black)

    # 경계를 강조합니다.
    thresh_combined = cv2.threshold(mask_combined, 150, 255, cv2.THRESH_BINARY)[1]

    # 경계선을 찾습니다.
    contours_combined, hierarchy = cv2.findContours(thresh_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 노란색과 검은색이 함께 이루어진 삼각형을 찾습니다.
    triangles_combined = []
    for contour in contours_combined:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if len(approx) == 3 and cv2.contourArea(approx) > 300:
            triangles_combined.append(approx)
            

    # 삼각형을 그립니다.
    for triangle in triangles_combined:
        cv2.drawContours(frame, [triangle], 0, (0, 255, 0), 2)
        for i in [triangle]:
            a=0
            for j in i:
                cv2.circle(frame, tuple(j[0]), 1, (255,0,0), -1)
                x,y=j[0]
                
                a=a+1
                if a == 1:
                    print("1번",x,y)
                    x1,y1=x,y
                if a == 2:
                    print("2번",x,y)
                    x2,y2=x,y
                if a == 3:
                    print("3번",x,y)
                    x3,y3=x,y
                
                
                
        M = cv2.moments(triangle)
        if M["m00"] != 0:
            # 삼각형 중심을 계산합니다.
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            print(cx,cy)
            dx1=abs(cx-x1)
            dx2=abs(cx-x2)
            dx3=abs(cx-x3)
            dy1=abs(cy-y1)
            dy2=abs(cy-y2)
            dy3=abs(cy-y3)
            if dx1 > dx2:
                dx = dx2
                if dx < dx3:
                    if dy1 > dy3:#2,1  x2#윗점 y1#밑점
                        rx=(x1+x3)/2
                        ry=(y1+y2)/2                  
                        w=abs((x1-x3)/2)
                        h=abs((y2-y1)/2)
                    if dy1 <= dy3:#2,3 x2#윗점 y3#밑점
                        rx=(x1+x3)/2
                        ry=(y3+y2)/2  
                        w=abs((x1-x3)/2)
                        h=abs((y2-y3)/2)
                        
                if dx >= dx3:
                   
                    if dy1 > dy2:#3,1  x3#윗점 y1#밑점
                        rx=(x1+x2)/2
                        ry=(y1+y3)/2
                        w=abs((x1-x2)/2)
                        h=abs((y3-y1)/2)
                    if dy1 <= dy2:#3,2  x3#윗점 y2#밑점
                        rx=(x1+x2)/2
                        ry=(y3+y2)/2
                        w=abs((x1-x2)/2)
                        h=abs((y3-y2)/2)
                
            if dx1 <= dx2:
                dx = dx1
                if dx < dx3:
                 
                    if dy3 > dy2:#1,3  x1#윗점 y3#밑점
                        rx=(x3+x2)/2
                        ry=(y3+y1)/2
                        w=abs((x2-x3)/2)
                        h=abs((y1-y3)/2)
                    if dy3 <= dy2:#1,2  x1#윗점 y2#밑점
                        rx=(x3+x2)/2
                        ry=(y1+y2)/2
                        w=abs((x3-x2)/2)
                        h=abs((y3-y1)/2)
                if dx >= dx3:
         
                    if dy1 > dy2:#3,1  x3#윗점 y1#밑점
                        rx=(x1+x2)/2
                        ry=(y1+y3)/2
                        w=abs((x1-x3)/2)
                        h=abs((y2-y1)/2)
                    if dy1 <= dy2:#3,2  x3#윗점 y2#밑점
                        rx=(x1+x2)/2
                        ry=(y3+y2)/2
                        w=abs((x1-x3)/2)
                        h=abs((y2-y3)/2)
            d=d+1
            x = float(rx)/w1
            y = float(ry)/h1
            w = float(w)/(w1/2)
            h = float(h)/(h1/2)
            print(x," ",y," ",w," ",h)
            cv2.imwrite('C://Users//82102//Desktop//label//%s (%d).jpg'%(answer,d),dst)
            flie = "C://Users//82102//Desktop//label//%s (%d).txt"%(answer,d)
            f = open(flie, 'w')
            data=str(answer)
            f.write(data)
            f.write(' ')
            data=str(x)
            f.write(data)
            f.write(' ')
            data=str(y)
            f.write(data)
            f.write(' ')
            data=str(w)
            f.write(data)
            f.write(' ')
            data=str(h)
            f.write(data)
            cv2.circle(frame,(cx,cy), 1, (255,0,0), -1)

    # 결과를 출력합니다.
    cv2.imshow('frame2', mask_combined )
    cv2.imshow('frame', frame )
    if cv2.waitKey(1) == ord('q'):
        break
while answer == 2:
    # 카메라에서 프레임 읽기
    ret, frame = capture.read()
    dst = frame.copy()
    # 색공간 변경 (BGR → HSV)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 빨간색 영역 추출
    lower_red = np.array([160, 100, 100])
    upper_red = np.array([179, 255, 255])
    red_mask = cv2.inRange(hsv_frame, lower_red, upper_red)

    # 빨간색 원 외곽선 찾기
    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        # 가장 큰 외곽선 찾기
        c = max(contours, key=cv2.contourArea)
        # 외곽선 내부를 검정색으로 채우기
        mask = np.zeros_like(red_mask)
        cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
        # 외곽선 내부의 흰색 영역 추출
        white_mask = cv2.bitwise_and(frame, frame, mask=mask)
        white_mask = cv2.cvtColor(white_mask, cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(white_mask, 10, 255, cv2.THRESH_BINARY)

        # 원 그리기
        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)
        if radius > 10:
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            x = float(x)/w1
            y = float(y)/h1
            w = float(radius)/(w1/2-10)
            h = float(radius)/(h1/2-10)
            print(x," ",y," ",w," ",h)
            d=d+1
            #라벨 경로
            flie = "C://Users//82102//Desktop//label/%s (%d).txt"%(answer,d)
            cv2.imwrite('C://Users//82102//Desktop//label//%s (%d).jpg'%(answer,d),dst)
            f = open(flie, 'w')
            data=str(answer)
            f.write(data)
            f.write(' ')
            data=str(x)
            f.write(data)
            f.write(' ')
            data=str(y)
            f.write(data)
            f.write(' ')
            data=str(w)
            f.write(data)
            f.write(' ')
            data=str(h)
            f.write(data)

    # 결과 출력
    cv2.imshow("Red Circle Detection", frame)
    #cv2.imshow("Red Circle Detection2", white_mask)
    # 종료 조건
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
while answer == 3:
    # 이미지 읽기
    ret, frame = capture.read()
    dst = frame.copy()
    frame1=frame
    

    # 파란색 범위 지정
    lower_blue =np.array([99, 50, 50]) 
    upper_blue = np.array([120, 255, 255])
    
    # 이미지를 HSV 형식으로 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 파란색 범위에 해당하는 부분을 추출
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    cv2.imshow('frame1', mask_blue )
    # 추출한 이미지에서 사각형 찾기
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour_blue in contours_blue:
        x, y, w, h = cv2.boundingRect(contour_blue)

        # 사각형 내부의 흰색 물체 추출
        mask_rect = np.zeros_like(mask_blue)
        cv2.drawContours(mask_rect, [contour_blue], 0, (255, 255, 255), -1)
        mask_rect = cv2.bitwise_and(mask_rect, mask_blue)
        white_objects, _ = cv2.findContours(mask_rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for white_object in white_objects:
            if cv2.contourArea(white_object) > 50:
                x_w, y_w, w_w, h_w = cv2.boundingRect(white_object)
               
        # 사각형 표시
        if w > 50 and h > 50:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.circle(frame,(x,y), 1, (255,250,0), -1)
            rx=(2*x+w)/2
            ry=(2*y+h)/2
            w=w/2
            h=h/2
            x = float(rx)/w1
            y = float(ry)/h1
            w = float(w)/(w1/2)
            h = float(h)/(h1/2)
            d=d+1
            #라벨 경로
            flie = "C://Users//82102//Desktop//label/%s (%d).txt"%(answer,d)
            cv2.imwrite('C://Users//82102//Desktop//label//%s (%d).jpg'%(answer,d),dst)
            f = open(flie, 'w')
            data=str(answer)
            f.write(data)
            f.write(' ')
            data=str(x)
            f.write(data)
            f.write(' ')
            data=str(y)
            f.write(data)
            f.write(' ')
            data=str(w)
            f.write(data)
            f.write(' ')
            data=str(h)
            f.write(data)
    
    print("인식")


capture.release()#반복 끝나면 프로그램 끝내기

cv2.destroyAllWindows()
cv2.waitKey()

                       
