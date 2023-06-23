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


img = inpaint_text("./zz2.png", pipeline)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

cv2.imshow("thresh", thresh)

mor_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (3, 3), iterations=3)
mor_img = 255 - mor_img

contours, hierarchy = cv2.findContours(mor_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
cv2.imshow("img", img)
cv2.waitKey(0)
