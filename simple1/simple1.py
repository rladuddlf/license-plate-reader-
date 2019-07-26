# !/usr/bin/etc python

import cv2
import numpy as np
import pytesseract
import sys, os
import threading
import re
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'

'''
입력된 파일 이미지의 번호판을 인식해서 번호판 글자를 출력해주는 Recognition 클래스이다
<클래스내 함수 list>
checkTime()
checkFile()
checkExtention()
checkSize()
checkVolume()
checkColor()
ExtractNumber()
checkOutput()
'''


class Recognition:
    count = 0  # 프로그램이 시작한지 '몇 초' 인지에 대한 멤버변수
    file_path = ""  # 입력된 파일의 절대경로
    real_result = ""  # 결과로 나온 값
    '''
    <checkTime>
    스레드를 이용해서 프로그램이 시작하고 10초가 지났는지 판별하는 함수이다.
    클래스내 프로그램 종료 부분에서 타이머 스레드를 끝낸다.
    10초동안 스레드가 안끝났다면 플랫폼 자체 오류로 판단 후 종료한다.
    '''

    def checkTime(self):
        # 스레드의 타이머를 이용해 checkTime 함수를 1초마다 재귀호출했다.
        timer = threading.Timer(1, recogtest.checkTime)
        timer.start()
        # if quit():
        #   timer.cancel()
        #   quit => '프로그램이 끝난다면'을 구현하려 했으나 무한루프에 빠졌다. * 검사 보고서의 1.1.1의 1)참조
        # 따라서 다른 함수에서 종료하는 부분마다 count 멤버변수 값에 -1을 주어서
        # 이 함수에서 count 값을 -1로 받는다면 타이머 스레드가 멈추고 프로그램이 종료되게 했다. * 검사 보고서의 1.1.2의 2)참조
        if self.count == -1:
            timer.cancel()
            quit()
        self.count += 1
        print(self.count, "초")
        # 만약 10초가 됐다면 타이머 스레드를 종료한다.
        if self.count == 10:
            # <Unit_Test> 10초까지 잘 실행되는지 보기 위해 결과함수 주석처리 후 실행해 보았다.
            # print("10초 끝") </Unit_Test> * 검사 보고서의 1.1.1의 2)참조
            print("플랫폼 자체 오류입니다. 재실행 해주세요. error code: 0")
            timer.cancel()
            quit()

    '''
    <checkFile>
    파일의 갯수를 검사하는 함수.
    '''

    def checkFile(self):
        '''
        원래 이미지 파일의 존재여부 검사를 checkFile에서 진행하려 하였으나
        sys.argv를 통해서는 파일 존재여부를 알기가 어려워서 밑에 checkSize에서 실시하였다.
        처음에 생각한 전략은 if sys.argv[1] != None 의 형태였다.
        하지만 해당 방법은 sys.argv 배열에 값이 있는지만 판단하기에 정말 그 파일이 있는지는 알 수가 없다.
        따라서 checkFile에서는 파일의 갯수만 검사한다.
        '''
        if len(sys.argv) > 2:  # 2개 이상 입력하면 프로그램 종료
            # <Unit_Test> print(sys.argv) </Unit_Test> * 검사 보고서의 1.2.1의 1)참조
            print(sys.argv)
            print("파일을 1개만 입력해주세요. error code: 1")
            print("프로그램을 종료합니다.")
            self.count = -1
            quit()
        elif len(sys.argv) == 1:  # 입력을 안하면 프로그램 종료
            # <Unit_Test> print(sys.argv) </Unit_Test> * 검사 보고서의 1.2.1의 2) 참조
            print(sys.argv)
            print("파일을 입력해주세요 error code: 1")
            print("프로그램을 종료합니다.")
            self.count = -1
            quit()
        '''elif len(sys.argv) == 2:
            print(sys.argv)
            print("1개만 잘 입력하였습니다.")
            print("그 다음으로 넘어갑니다.")
            print()'''
        '''
        <Unit_Test> * 검사 보고서의 1.2.1의 3) 참조
        elif len(sys.argv) == 2:
            print(sys.argv)
            print("1개만 잘 입력하였습니다.")
            print("그 다음으로 넘어갑니다.")
            print()
        </Unit_Test>
        '''

    '''
    <checkExtention>
    파일 확장자를 검사하는 코드이다.
    checkFile 함수 다음에 호출되는 함수인 만큼 파일이 1개만 들어온다는 전제 하에 돌아간다.
    '''

    def checkExtention(self):
        # 사용자가 입력한 파일명 멤버변수에 저장
        self.file_path = sys.argv[1]
        # <Unit_Test> print(sys.argv[1]) </Unit_Test> * 검사 보고서의 1.3.1의 1) 참조
        # . 단위로 split해서 배열에 저장
        split_str = self.file_path.split('.')
        split_str_length = len(split_str)
        # 배열 마지막 칸이 확장자
        # exe_name = split_str[split_str_length] * 검사 보고서의 1.3.1의 2) 참조
        '''
        순간 R과 헷갈려서 index가 배열 길이일때 마지막인줄 알았다.
        index 오류인지 모르고 파일 경로가 들어왔는지 부터 split 됐는지 보았다.
        '''
        # <Unit_Test> print(split_str[0]) </Unit_Test> * 검사 보고서의 1.3.1의 3) 참조
        # <Unit_Test> print(split_str[1]) </Unit_Test> * 검사 보고서의 1.3.1의 3) 참조
        exe_name = split_str[split_str_length - 1]
        # if exe_name == 'jpg' or exe_name == 'jpeg' or exe_name == 'png':
        # 웬만하면 조건을 3개이상 쓰지 않기로 했다.

        # 확장자가 jpg, jpeg, png 라면
        if exe_name == 'jpg' or exe_name == 'jpeg':
            a = 1;
            # <Unit_Test> print('이것의 확장자는 : ' + exe_name)
            # print('옳은 확장자이다') </Unit_Test> * 검사 보고서의 1.3.2의 1) 참조
        elif exe_name == 'png':
            b = 2;
            # <Unit_Test> print('이것의 확장자는 : ' + exe_name)
            # print('옳은 확장자이다.') </Unit_Test> * 검사 보고서의 1.3.2의 1) 참조
        else:  # 셋다 아니면 오류출력 후 종료
            # <Unit_Test> print('이것의 확장자는 : ' + exe_name) </Unit_Test> * 검사 보고서의 1.3.2의 2) 참조
            print("표준 확장자가 아닙니다. error code: 1")
            print("프로그램을 종료합니다.")
            self.count = -1
            quit()

    '''
    <checkSize>
    파일 이미지의 가로X세로 사이즈를 구해서 예외처리 하는 함수이다.
    또한 파일의 존재여부를 확인한다.
    '''

    def checkSize(self):
        try:
            # 파일 이미지를 Image 객체에 저장
            im = Image.open(self.file_path)
            # 이미지 사이즈를 변수에 저장
            file_length = im.size
            # <Unit_Test> print("(가로,세로):", file_length) </Unit_Test> * 검사 보고서의 1.4.2  참조
            # 가로, 세로 변수 나누기
            (img_width, img_height) = file_length
            # <Unit_Test> print(img_width) </Unit_Test> * 검사 보고서의 1.4.2  참조
            # <Unit_Test> print(img_height) </Unit_Test> * 검사 보고서의 1.4.2  참조
            # 파일크기가 (150px, 100px) 이상인 것만 정상출력
            if img_width >= 150 and img_height >= 100:
                c = 3;
                # <Unit_Test> print("사이즈 OK") </Unit_Test> * 검사 보고서의 1.4.3의 1) 참조
            else:
                print("표준 사이즈가 아닙니다. error code: 1")
                print("프로그램을 종료합니다.")
                self.count = -1
                quit()

            # 파일 비율 안맞는것 에러처리
            if img_width < img_height*3 or img_width > img_height*6:
                print("이미지의 비율이 맞지 않습니다. error code: 1")
                print("프로그램을 종료합니다.")
                self.count = -1
                quit()

        except FileNotFoundError:  # file_path가 존재하는지 'FileNotFoundError'를 통해 검사.지금까지 입력 오류 검사들은 String 검사이기에 이번 함수에서 파일의 존재여부에 따라 Image.open을 한다.
            # <Unit_Test> print(sys.argv) </Unit_Test> * 검사 보고서의 1.4.1 참조
            print("파일을 찾을 수 없습니다. error code: 1")
            print("프로그램을 종료합니다.")
            self.count = -1
            quit()

    '''
    <checkVolume>
    파일의 용량을 구해서 예외처리 하는 함수이다.
    '''

    def checkVolume(self):
        # 파일 용량 구하기(바이트)
        file_size = os.path.getsize(self.file_path)
        # <Unit_Test> print(file_size) </Unit_Test> * 검사 보고서의 1.5.1의 1) 참조
        # KB 로 변환
        # <Unit_Test> file_size_kb = file_size / 1024
        # print("파일 용량 : ", file_size_kb, "KB") </Unit_Test> * 검사 보고서의 1.5.1의 2) 참조
        # 파일 사이즈 제한 초과(1MB == 1048576B)시 오류출력 후 종료
        if file_size > 1048576:
            # <Unit_Test> print('파일 용량 제한 초과') </Unit_Test> * 검사 보고서의 1.5.2 참조
            print("파일 용량 제한 초과입니다. error code: 1")
            print("프로그램을 종료합니다.")
            self.count = -1
            quit()

    '''
    <checkColor>
    이미지의 컬러 평균 값을 구하는 함수
    '''

    def checkColor(self):
        '''* 검사 보고서의 1.6.6 참조
        설계 초기에는 from sklearn.cluster import KMeans의 kmeans와 numpy의 histogram을 사용하여
        많이 존재하는 상위 5가지 색을 추출해 각각의 색이 조건에 맞는지 비교하는 방식을 고민하였으나
        이미지에서 컬러 추출만 하는 코드를 작성해 돌려본 결과, 작은 사진임에도 아주 느렸다.
        (64kb 사진이 약 6초~7초가 걸림)
        그래서 각 픽셀의 RGB값을 평균값내는 방식으로 설계하게 되었다.
        '''
        # <Unit_Test> image_path = "./img/colorimg1.jpg" </Unit_Test>* 검사 보고서의 1.6.1 참조
        # <Unit_Test> img = cv2.imread(image_path) </Unit_Test>
        img = cv2.imread(self.file_path)#이미지 읽어오기
        # <Unit_Test> img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) </Unit_Test>* 검사 보고서의 1.6.2 참조
        h, w, channels = img.shape# 이미지 데이터의 행, 열, 채널
        img = img.reshape((w * h, 3)) # img는 원래 h,w,channels의 3차원이지만 계산하기 쉽도록 w*h, 3의 2차원으로 바꾼다.

        area = float(w * h)#이미지의 넓이
        r = 0;#r, g, b값 0으로 초기화
        g = 0;
        b = 0;
        for i in range(0, w * h): # 전체 합 계산
            b = b + img[i][0]#BGR이라서 인덱스가 B가 0, G가 1, R이 2이다.
            g = g + img[i][1]
            r = r + img[i][2]

        r = float(r) / area # 전체 합을 픽셀의 수로 나눈다.
        g = float(g) / area
        b = float(b) / area

        # <Unit_Test> print("가장 많이 차지하는 색 R: {0}, G: {1}, B: {2}".format(r, g, b)) </Unit_Test>* 검사 보고서의 1.6.3 참조
        # <Unit_Test> print(min(r,g,b)) </Unit_Test>* 검사 보고서의 1.6.4 참조
        # <Unit_Test> print(max(r,g,b)) </Unit_Test>
        if (max(r, g, b)  > min(r, g, b) * 1.2):  # 다른 색이 너무 튀는 경우 종료
            print('입력 이미지의 컬러가 조건에 맞지 않습니다.')#에러 출력
            self.count = -1#타이머 멈추기
            quit()#프로그램 종료
        #if문을 만족하지 않으면 다음 과정(ExtractNumber)으로 넘어간다.

    '''
    <ExtractNumber>
    이미지에서 번호판의 글자 영역만 추출해 pytesseract에 적용하는 함수
    '''

    # <Unit_Test></Unit_Test>
    def ExtractNumber(self):
        '''
        설계 단계에서 정확성을 위해
        (threshold 적용 -> erode -> 글자 추출)에서
        (contours 찾기 -> 이미지 자르기 -> threshold적용 -> erode -> 글자 추출)로 결정되었다.
        '''
        img = cv2.imread(self.file_path, cv2.IMREAD_COLOR) # conrours를 찾기 위해 gray -> blur -> canny과정을 거친다.
        # <Unit_Test> Number = './img/eximg2.jpg' </Unit_Test>* 검사 보고서의 1.7.1 참조
        # <Unit_Test> img = cv2.imread(Number, cv2.IMREAD_COLOR) </Unit_Test>
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # BGR에서 GRAY로 바꿈.
        # <Unit_Test> cv2.imwrite('./img/gray.jpg', img2) </Unit_Test>* 검사 보고서의 1.7.2 참조
        blur = cv2.GaussianBlur(gray_img, (3, 3), 0) # 노이즈 제거를 위해 사용. 필터는 (3,3), 0은 sigmaX값
        # <Unit_Test> cv2.imwrite('./img/blur.jpg', blur) </Unit_Test>* 검사 보고서의 1.7.3 참조
        canny = cv2.Canny(blur, 150, 200) # canny edge detection, minimum thresholding value는 150, maximum thresholding value는 200
        # <Unit_Test> cv2.imwrite('./img/canny.jpg', canny) </Unit_Test>* 검사 보고서의 1.7.4 참조
        __, contours, __ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # <Unit_Test> cnts, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) </Unit_Test>* 검사 보고서의 1.7.5 참조
        # canny에서 contours를 찾는다. cv2.RETR_TREE 이미지에서 모든 contour추출,
        # cv2.CHAIN_APPROX_SIMPLE 윤곽선에 외접하는 사각형이 가지는 4개의 꼭짓점으로 contour 근사.

        height, width, __ = img.shape # 이미지 데이터의 행, 열, 채널
        # <Unit_Test> height, width, channels = copy_img.shape </Unit_Test>* 검사 보고서의 1.7.6 참조
        # <Unit_Test> print("height = {0}, width = {1}".format(height,width)) </Unit_Test>

        rect_x = [] # 조건에 맞아 글자로 판정되는 contour에 외접하는 사각형의 좌상단 x, y값과 w, h값이 저장된다.
        rect_y = []
        rect_w = []
        rect_h = []

        # <Unit_Test> loop = len(contours) </Unit_Test>* 검사 보고서의 1.7.7 참조
        # <Unit_Test> print(loop) </Unit_Test>

        for i in range(len(contours)):
         # <Unit_Test> for i in range(loop): #len을 변수에 넣고 돌려보았다. len(contours)와 결과는 같다.</Unit_Test>
            cnt = contours[i]#contour 중 i번째를 cnt가 받는다.
            x, y, w, h = cv2.boundingRect(cnt) # 인자로 받은 contour에 외접하는 사각형의 좌상단 x,y와 너비, 높이 값을 리턴한다.
            rect_ratio = float(w * h) / (width * height) # 전체 이미지에서 이 contours가 차지하는 넓이 비율
            aspect_ratio = float(w) / h  # 글자인 경우, 대개 높이가 너비보다 크다. 높이 대비 너비가 가지는 비율

            if (aspect_ratio >= 0.2) and (aspect_ratio <= 1.0) and (rect_ratio > 0.03) and (rect_ratio < 0.1) and (
                        h < height * 0.95) and(h > height*0.5):
                #조건 :  세로 대비 가로 비율이 0.2~1.0, 전체 이미지에서 차지하는 넓이가 0.03~0.1, 높이는 전체 이미지의 0.5~0.95
                # <Unit_Test>cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1) </Unit_Test>* 검사 보고서의 1.7.8 참조
                rect_x.append(x) # append로 list에 추가
                rect_y.append(y)
                rect_w.append(w)
                rect_h.append(h)
                # <Unit_Test> print(rect_ratio) </Unit_Test>* 검사 보고서의 1.7.9 참조
                # <Unit_Test> print(aspect_ratio) </Unit_Test>
                # <Unit_Test> print("==============") </Unit_Test>

         # <Unit_Test> print(len(rect_x)) </Unit_Test>* 검사 보고서의 1.7.10 참조
        if (len(rect_x) > 9 and len(rect_x) < 19): # 이상적인 반환 결과는 14개의 길이를 가진다.
            x1 = min(rect_x)
            y1 = min(rect_y)
            x2 = max(rect_x) + max(rect_w)
            y2 = max(rect_y) + max(rect_h)
            # <Unit_Test> print("x1:{0} y1:{1} x2:{2} y2:{3}".format(x1,y1,x2,y2)) </Unit_Test>* 검사 보고서의 1.7.11 참조

            roi = img[y1:y2, x1:x2] # 이미지 자르기
            # <Unit_Test> cv2.imwrite('./img/test.jpg', roi) </Unit_Test>* 검사 보고서의 1.7.12 참조
            # <Unit_Test> cv2.imwrite('./img/snake.jpg', img) </Unit_Test>* 검사 보고서의 1.7.8 참조

            border_x = int((x2 - x1) * 0.1)#좌우에 추가될 border의 크기
            border_y = int((y2 - y1) * 0.5)#위아래에 추가될 border의 크기
            constant = cv2.copyMakeBorder(roi, border_y, border_y, border_x , border_x , cv2.BORDER_CONSTANT, value=[255, 255, 255])
            # <Unit_Test> cv2.imwrite('./img/test2.jpg', constant) </Unit_Test>* 검사 보고서의 1.7.12 참조
            # <Unit_Test> cv2.imwrite('./img/snake.jpg', img) </Unit_Test>* 검사 보고서의 1.7.8 참조
            # roi에 위, 아래 border_y(px) 왼쪽, 오른쪽에 border_x(px) 씩 cv2.BORDER_CONSTANT로 value=[255,255,255]색상의 테두리 추가

            plate_gray = cv2.cvtColor(constant, cv2.COLOR_BGR2GRAY)
            __, th_plate = cv2.threshold(plate_gray, 150, 255, cv2.THRESH_BINARY)
            # <Unit_TestUnit_Test> ret, th_plate = cv2.threshold(plate_gray, 100, 255, cv2.THRESH_BINARY) </Unit_Test>* 검사 보고서의 1.7.13 참조
            # <Unit_Test> cv2.imwrite('./img/plate_th.jpg', th_plate) </Unit_Test>
            # plate_gray에 픽셀 문턱값은 150, 문턱값보다 클 때는 255를 적용한다.
            # 흑백이미지여야 하므로 문턱값보다 작으면 0이 적용되도록 cv2.THRESH_BINARY

            kernel = np.ones((3, 3), np.uint8) # 커널 생성. 3x3에 1로 채워진 매트릭스
            er_plate = cv2.erode(th_plate, kernel, iterations = 1)
            # th_plate에 erosion을 하는데 사용하는 kernel은 위에 만든 3x3커널. iteration은 반복 횟수를 의미한다. 1번만 한다.
            # <Unit_Test> er_plate = cv2.dilate(th_plate, kernel, iterations=1)* 검사 보고서의 1.7.14 참조
            # dilate했더니 오히려 가늘어진다. dilate는 백색이 커지고 검은색이 작아짐 </Unit_Test>
            # <Unit_Test> cv2.imwrite('./img/er_plate.jpg', er_plate) </Unit_Test>* 검사 보고서의 1.7.15 참조
            result = pytesseract.image_to_string(er_plate, lang = 'kor')
            # pytesseract의 image_to_string함수로 er_plate에 tesseract ocr을 적용한 결과를 result에 넣는다.
            # <Unit_Test> result = pytesseract.image_to_string(Image.open('./img/er_plate.jpg'), lang='kor') </Unit_Test>
            self.real_result = result.replace(" ", "") # 반환되는 글자들 사이에 " "을 없애준다.
            self.checkOutput() # 출력 에러 검사 시행
            self.count = -1 # time세는걸 멈춘다.

            return "인식 결과 번호판은 " + self.real_result + "입니다."#검사 결과 반환
            # <Unit_Test> self.count = -1 </Unit_Test>* 검사 보고서의 1.7.16 참조
            # <Unit_Test> sys.exit() </Unit_Test>
        else:#이미지 내에 번호판 글자가 없다고 판단된 경우
            self.count = -1#타이머 멈추기
            return "표준 번호판 이미지가 아닙니다. error code: 2"
            # <Unit_Test> self.count = -1 </Unit_Test>
            # <Unit_Test> sys.exit() </Unit_Test>

    '''
    <checkOutput>
    출력 메시지가 '숫자,숫자,한글,숫자,숫자,숫자,숫자'의 형태인지 검사하는 함수.
    '''

    def checkOutput(self):
        '''
        기존에 생각한 전략은 반복문을 돌려 각각의 원소의 타입이 int인지 str인지 먼저 구분하는거 였으나
        ExtractNumber()가 실행되면서 pytesseract.image_to_string()에 의해 모두 str으로 바뀌게된다.
        그래서 아스키값과 유니코드를 통해 검사하는 전략으로 바꾸었다.
        '''
        if len(self.real_result) != 7:  # 문자열이 7개가 아니면 종료.
            # <Unit_Test> print("번호판 인식 결과: " + self.real_result) </Unit_Test> * 검사 보고서 1.8.1 참조
            print("일곱 글자가 아닙니다. error code: 1")
            print("출력 오류입니다. 프로그램을 종료합니다.")
            self.count = -1
            quit()
        else:  # 문자열이 7개면 각각의 원소들에 대하여 출력 에러 검사를 실시한다.
            for i in range(0, 7):
                # <Unit_Test> print(ord(self.real_result[i])) #각 출력 원소의 아스키값 확인. </Unit_Test> * 검사 보고서 1.8.2 참조
                if i != 2:  # 한글은 아스키값이 없으므로 3번째 문자열은 제외. 나머지 문자열들이 숫자인지 검사.
                    if ord(self.real_result[i]) < 48 or ord(
                            self.real_result[i]) > 57:  # 0~9의 아스키값은 48~57이므로 이에 해당 안되면 종료.
                        print("출력이 '숫자,숫자,한글,숫자,숫자,숫자,숫자'의 형태가 아닙니다. error code: 2")
                        print("출력 오류입니다. 프로그램을 종료합니다.")
                        self.count = -1
                        quit()

                else:  # 3번째 문자열이 한글인지 확인
                    '''
                    한글은 아스키값이 없어서 유니코드를 활용하여 3번째 문자열이 한글인지 확인하려고 하였다.
                    3번째 문자열을 유니코드로 인코딩한 후, 해당 타입을 통해 확인하려 하였으나 인코딩을 한 시점에서는 모두 utf-8로 인코딩이 되어 타입을 비교하는 것은 무의미하였다.
                    하지만 이 과정에서 힌트를 얻어 정규식을 통해 한글을 구분할 수 있을 것 같았다.
                '''
                    korean = self.real_result[i]
                    compileKorean = re.compile(u'[\u3130-\u318F\uAC00-\uD7A3]+')  # 한글과 띄어쓰기를 제외한 모든 글자.
                    isKorean = compileKorean.findall(korean)  # 인자로 들어온 korean에서 패턴[\u3130-\u318F\uAC00-\uD7A3]을 만족하는 문자열을 리스트로 반환. 해당 패턴은 한글 코드 범위를 나타냄
                    # <Unit_Test> print(korean) # 3번째 문자열 확인해보기 </Unit_Test> * 검사 보고서 1.8.3 참조
                    # <Unit_Test> print(compileKorean) # 패턴 확인해보기 </Unit_Test> * 검사 보고서 1.8.3 참조
                    # <Unit_Test> print(isKorean) # 패턴에 맞는 문자(한글) 확인해보기 </Unit_Test> * 검사 보고서 1.8.3 참조
                    if len(isKorean) == 0:  # 리스트에 아무것도 없으면 한글이 없다는 뜻이므로 오류로 판단.
                        print("3번째 문자열이 한글이 아닙니다.")
                        print("출력이 '숫자,숫자,한글,숫자,숫자,숫자,숫자'의 형태가 아닙니다. error code: 2")
                        print("출력 오류입니다. 프로그램을 종료합니다.")
                        self.count = -1
                        quit()
                    '''
                    <Unit_test> * 검사 보고서 1.8.3 참조
                    else:
                        print(isKorean)
                        print("한글입니다.")
                    </Unit_Test>
                    '''

                    '''
                    <Unit_Test> * 검사 보고서 1.8.4 참조
                    print("==================한글 테스트====================")
                    korean = '뷁홡와워웨왜굳욕샮'
                    asdf = re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', korean)
                    hanCount = len(re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', korean))
                    if hanCount > 0:
                        print(korean + "(" + str(asdf) + ")" + "는 한글")
                    else:
                        print(korean + "(" + str(asdf) + ")" + "는 한글이 아님")

                    english = u'english'
                    asdf2 = re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', english)
                    hanCount2 = len(re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', english))
                    if hanCount2 > 0:
                        print(english + "(" + str(asdf2) + ")" + "는 한글")
                    else:
                        print(english + "(" + str(asdf2) + ")" + "는 한글이 아님")

                    hangul = 'asdf,ㅁㄴㅇㄹ,1234,!@$#'
                    asdf3 = re.compile('[\u3130-\u318F\uAC00-\uD7A3]+')
                    result = asdf3.sub('', hangul)
                    result2 = asdf3.findall(hangul)
                    print("<" + hangul + ">" + "를 분류하면 " + result +"이고 여기서 한글은 " + str(result2) + "이다.")
                    print()
                    </Unit_Test>
                    '''

    def Unicode(self):
        print("==================한글 테스트====================")
        korean = '뷁홡와asdf 워웨왜굳욕 샮'
        korean.encode('utf-8')
        print(korean)
        #re.compile(u'[\u3130-\u318F\uAC00-\uD7A3]+')
        #print(type(re.compile(u'[\u3130-\u318F\uAC00-\uD7A3]+')))
        #print(re.compile(korean))
        asdf = re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', korean)
        asdf2 = re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]', korean)
        print(asdf)
        print(asdf2)
        print(len(asdf))
        print(len(asdf2))
        hanCount = len(re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', korean))
        if hanCount > 0:
            print(korean + "(" + str(asdf) + ")" + "는 한글")
        else:
            print(korean + "(" + str(asdf) + ")" + "는 한글이 아님")

        english = u'english'
        asdf2 = re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', english)
        hanCount2 = len(re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', english))
        if hanCount2 > 0:
            print(english + "(" + str(asdf2) + ")" + "는 한글")
        else:
            print(english + "(" + str(asdf2) + ")" + "는 한글이 아님")

        hangul = 'asdf,ㅁㄴㅇㄹ,1234,!@$#'
        asdf3 = re.compile('[\u3130-\u318F\uAC00-\uD7A3]+')
        result = asdf3.sub('', hangul)
        result2 = asdf3.findall(hangul)
        #print(asdf3)
        print("<" + hangul + ">" + "를 분류하면 " + result + "이고 여기서 한글은 " + str(result2) + "이다.")
        print()

recogtest = Recognition()
# recogtest.checkTime()
recogtest.checkFile()
recogtest.checkExtention()
recogtest.checkSize()
recogtest.checkVolume()
recogtest.checkColor()
result = recogtest.ExtractNumber()
print(result)
#recogtest.Unicode()
#print(sys.argv)