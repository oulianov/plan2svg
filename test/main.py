import cv2
import random
import keras_ocr
import math
import numpy as np


def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2) / 2)
    y_mid = int((y1 + y2) / 2)
    return (x_mid, y_mid)


pipeline = keras_ocr.pipeline.Pipeline()


def inpaint_text(img_path, pipeline):
    # read image
    img = keras_ocr.tools.read(img_path)
    # generate (word, box) tuples
    prediction_groups = pipeline.recognize([img])
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1]
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]

        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)

        thickness = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255, thickness)
        img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)

    return img


def find_contours():
    mor_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (3, 3), iterations=3)
    # mor_img = 255 - mor_img

    contours, hierarchy = cv2.findContours(
        mor_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    # contours = np.vstack(contours)

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in sorted_contours[1:]:
        area = cv2.contourArea(c)
        if area > 10:
            print(area)
            cv2.drawContours(
                img,
                [c],
                -1,
                (
                    random.randrange(100, 255),
                    random.randrange(100, 255),
                    random.randrange(100, 255),
                ),
                3,
            )
            x, y, w, h = cv2.boundingRect(
                c
            )  # the lines below are for getting the approximate center of the rooms
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            cv2.putText(
                img,
                str(area),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

    cv2.imshow("mor_img", mor_img)


def find_contours_2():
    # Grayscale and Black and white
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # erode to remove small lines
    tresh = cv2.erode(thresh, np.ones((5, 5), np.uint8), iterations=2)

    result_fill = np.ones(img.shape, np.uint8) * 255
    result_borders = np.zeros(img.shape, np.uint8)

    # the '[:-1]' is used to skip the contour at the outer border of the image
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0][:-1]

    # fill spaces between contours by setting thickness to -1
    cv2.drawContours(result_fill, contours, -1, 0, -1)
    cv2.drawContours(result_borders, contours, -1, 255, 1)

    # xor the filled result and the borders to recreate the original image
    result = result_fill ^ result_borders

    # prints True: the result is now exactly the same as the original
    print(np.array_equal(result, img))

    cv2.imwrite("contours.png", result)
    cv2.imshow("mor_img", result)
    cv2.waitKey(0)


# Remove text
img = inpaint_text("./zz.jpg", pipeline)

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

kernel_size = 5
blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
cv2.imshow("blur", blur)

# Grayscale and Black and white
_, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
# erode to remove small lines
tresh = cv2.erode(thresh, np.ones((5, 5), np.uint8), iterations=2)

low_t = 50
high_t = 150
edges = cv2.Canny(tresh, low_t, high_t)
cv2.imshow("edges", edges)

rho = 3
theta = np.pi / 180
threshold = 15
min_line_len = 60
max_line_gap = 60
lines = cv2.HoughLinesP(
    edges,
    rho,
    theta,
    threshold,
    np.array([]),
    minLineLength=min_line_len,
    maxLineGap=max_line_gap,
)

# Draw the lines
if lines is not None:
    for i in range(0, len(lines)):
        l = lines[i][0]
        cv2.line(img, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

cv2.imshow("result", img)
cv2.waitKey(0)
