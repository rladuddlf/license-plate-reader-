import cv2

def image_color(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    w, h, channels = image.shape
    image = image.reshape((w * h, 3))

    area = float(w*h)
    r=0;g=0;b=0;
    for i in range(0,w*h):
        r = r + image[i][0]
        g = g + image[i][1]
        b = b + image[i][2]

    r = r / area
    g = g / area
    b = b / area

    print("가장 많이 차지하는 색 R: {0}, G: {1}, B: {2}".format(r, g, b))

    print(min(r, g, b))
    print(max(r, g, b))

    if (max(r, g, b) > min(r, g, b) * 1.2):  # 다른 색이 너무 튀는 경우
        print('입력 이미지의 컬러가 조건에 맞지 않습니다.')
    else:
        print('입력 이미지의 컬러가 조건 만족')

image_path = "colorCheck.png"
image_color(image_path)