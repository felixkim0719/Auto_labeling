# 자동 라벨링 프로그램

## 1. 라벨링이란

### 1.1 개요
라벨링이란 주어진 데이터에 정답지를 만들어주는 작업으로, 이 정답지를 **라벨**이라고 합니다. 딥러닝(Deep Learning)에서 **지도학습(Supervised Learning)**을 하는 경우, 주어지는 데이터에 대해 라벨이 필요하며, 부정확한 라벨로 학습할 경우 모델 성능이 떨어지기 때문에 **정확한 라벨링**이 매우 중요합니다.

## 2. 제작 동기
ROS 기반 자율주행 시스템에서 **표지판 학습을 위한 데이터 라벨링**은 중요한 작업입니다. 하지만 수작업으로 라벨링을 할 경우 시간이 많이 소요되므로, 이를 효율적으로 자동화하기 위해 **자동 라벨링 프로그램**을 개발하게 되었습니다. 이 프로그램은 표지판 인식 및 라벨링을 자동화하여 개발 시간을 크게 단축하는 것을 목표로 하고 있습니다.

## 3. 프로그램 설명

### 3.1 라벨링 대상
자동 라벨링 대상은 다음과 같은 표지판들입니다:
- 교차로 (Intersection)
- 건설 구역 (Construction)
- 철도 건널목 (Level Crossing)
- 주차 구역 (Parking)
- 터널 (Tunnel)

### 3.2 자동 라벨링 목적
자동 라벨링 프로그램은 다양한 표지판 객체를 빠르고 정확하게 검출하여 **머신러닝 모델 학습용 데이터**를 자동으로 생성하는 데 목적을 두고 있습니다.

## 4. 프로그램 구조 설명

### 4.1 초기 설정
- 카메라로 비디오 스트림을 캡처
- 프레임의 너비와 높이 설정
- `classes.txt` 파일에 라벨링할 클래스 정보 저장

### 4.2 표지판 검출 과정
1. **교차로 표지판 검출**
   - 흑백 이미지로 변환 후 Hough 변환을 통해 원을 검출하고, 위치와 크기를 기반으로 라벨 파일 생성
2. **공사장 & 터널 표지판 검출**
   - HSV 색상 공간에서 노란색과 검은색 영역을 결합해 경계를 강조하고 삼각형을 검출하여 라벨 파일 생성
3. **정지 표지판 검출**
   - 빨간색 영역을 추출하여 원을 검출하고 위치와 크기를 기반으로 라벨 파일 생성
4. **주차 표지판 검출**
   - 파란색 영역에서 사각형을 찾아 라벨 파일 생성
5. **프로그램 종료**
   - 카메라 및 모든 OpenCV 창 종료

## 5. 프로그램 소스 설명

### 5.1 교차로 표지판 검출 소스 설명
- **프레임 읽기**: `capture.read()`를 사용해 카메라로부터 프레임을 읽어옵니다.
- **그레이스케일 변환**: `cv2.cvtColor()`를 사용해 그레이스케일로 전환합니다.
- **Hough 변환**: `cv2.HoughCircles()`로 원을 검출하고, 좌표와 크기를 저장합니다.

### 5.2 공사장 & 터널 표지판 검출 소스 설명
- **HSV 변환 및 색상 추출**: HSV 색상 공간으로 변환 후 노란색과 검은색 영역을 추출하여 경계를 강조합니다.
- **삼각형 검출 및 저장**: 외곽선을 추출하고, 삼각형의 위치와 크기를 저장합니다.

## 6. 추가 활동

### 6.1 모델 성능 향상 기능
- 이미지의 **명도 조정** 및 **회전 변형** 기능을 추가하여 학습 데이터를 다양화하고 모델 성능을 향상시켰습니다.

### 6.2 개선 사항
- 향후 다양한 환경과 여러 표지판을 동시에 검출할 수 있는 **향상된 알고리즘**으로 개선할 예정입니다.

## 7. 소스 코드
궁금한 점이 있으시면 아래 메일로 주세요.
Doyoung Kim (felixkim0719@gmail.com)
