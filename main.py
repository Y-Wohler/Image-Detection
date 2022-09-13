import cv2
import numpy as np
import sys

def image_per(mask_map, inputImg):

    contours, heirarchy = cv2.findContours(mask_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        maxCnt = max(contours, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(maxCnt, True)
        approx = cv2.approxPolyDP(maxCnt, epsilon, True)
        approx = approx.tolist()
        img_edge = []
        for i in range(len(approx)):
            pnt = approx[i][0]
            img_edge.append(pnt)
            cv2.circle(inputImg, tuple(pnt), 7, (0, 255, 0), -1)
        if len(img_edge) == 4:
            distance = []
            for i in range(len(img_edge) - 1):
                dist = ((img_edge[i+1][0] - img_edge[i][0]) ** 2 + (img_edge[i+1][1] - img_edge[i][1]) ** 2) ** (1 / 2)
                distance.append(dist)
            last_distance = ((img_edge[-1][0] - img_edge[0][0]) ** 2 + (img_edge[-1][1] - img_edge[0][1]) ** 2) ** (1 / 2)
            distance.append(last_distance)
            map_h = int(min(distance))
            map_w = int(max(distance))

            input_pts = np.float32(img_edge)
            inputImg = cv2.circle(inputImg, (img_edge[0][0], img_edge[0][1]), 5, (255, 0, 0))
            output_pts = np.float32([[map_w, 0], [0, 0], [0, map_h], [map_w, map_h]])

            m = cv2.getPerspectiveTransform(input_pts, output_pts)
            warp_output = cv2.warpPerspective(inputImg, m, (map_w, map_h))
            return warp_output
        else:
            print("No Contours")
    else:
        print("Corners Unknown")
        return None


def positions():

    ### get blue circle

    image_mask = 0
    contours, heirarchy = cv2.findContours(image_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_contour = max(contours, key=cv2.contourArea)

    ## Get 3 corner coordinates of triangle ( marker )
    arcs = 0.02 * cv2.arcLength(img_contour, True)
    approx = cv2.approxPolyDP(img_contour, arcs, True)
    approx = approx.tolist()

    img_edge = []
    for pnt in approx:
        pnt = pnt[0]
        img_edge.append(pnt)

    distance = []
    for i in range(len(img_edge)-1):
        dist = ((img_edge[i][0] - img_edge[0][0]) ** 2 + (img_edge[i][1] - img_edge[0][1]) ** 2) ** (1 / 2)
        distance.append(dist)

    ### Analyse which one is the shortest edge of the triangle and accordingly decide pointer nose a112
    result_list = [xpos, ypos, ang]
    return result_list


def image_reader(img):
    #   ’’’ Processes input image and returns the marker position and bearing ’’’

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min, s_min, v_min = 36, 0, 0
    h_max, s_max, v_max = 86, 255, 255

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    image_mask = cv2.inRange(hsv_img, lower, upper)

    ### Morphological operation for filling the triangle
    thresh = 255 - image_mask
    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((11, 11), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    i = 1
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        crop = hsv_img[y:y + h, x:x + w]
        cv2.imwrite("new_develop.jpg".format(i), crop)
        i = i + 1
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bh_min, bs_min, bv_min = 100, 150, 0
    bh_max, bs_max, bv_max = 140, 255, 255

    blue_lower = np.array([bh_min, bs_min, bv_min])
    blue_upper = np.array([bh_max, bs_max, bv_max])
    blue_square = cv2.inRange(hsv_img, blue_lower, blue_upper)
    thresh = 255 - blue_square
    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((11, 11), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    hsv_img = cv2.cvtColor(hsv_img,cv2.COLOR_HSV2BGR)
    i=1
    for cntr in contours:
        bx, by, bw, bh = cv2.boundingRect(cntr)
        crop = hsv_img[by:by + bh, bx:bx + bw]
        cv2.imwrite("blueSquare.jpg".format(i), crop)
        i = i + 1
    cv2.circle(hsv_img, (int(bx+(bw/2)), int(by+(bh/2))), radius=4,color=(0,0,255), thickness=2)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rh_min, rs_min, rv_min = 160, 100, 20
    rh_max, rs_max, rv_max = 180, 255, 255

    red_lower = np.array([rh_min, rs_min, rv_min])
    red_upper = np.array([rh_max, rs_max, rv_max])
    red_circle = cv2.inRange(hsv_img, red_lower, red_upper)
    thresh = 255 - red_circle
    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((11, 11), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    hsv_img = cv2.cvtColor(hsv_img,cv2.COLOR_HSV2BGR)
    i = 1
    for cntr in contours:
        rx, ry, rw, rh = cv2.boundingRect(cntr)
        crop = hsv_img[ry:ry + rh, rx:rx + rw]
        cv2.imwrite("redCircle.jpg".format(i), crop)
        i = i + 1
    cv2.circle(hsv_img, (int(rx+(rw/2)),int(ry+(rh/2))), radius=4,color=(255,255,255), thickness=2)
    xbpos = bx+(bw/2)
    ybpos = by+(bh/2)
    xrpos = rx+(rw/2)
    yrpos = ry+(rh/2)

    blue_point = (xbpos,ybpos)
    red_point = (xrpos,yrpos)
    blue_angle = np.arctan2(*blue_point[::-1])
    red_angle = np.arctan2(*red_point[::-1])
    final_angle = np.rad2deg((red_angle-blue_angle)%(2*np.pi))

    return [(xbpos,ybpos),(xrpos,yrpos), final_angle]
    # -------------------------------------------------------------------------------


# Main program .
# -------------------------------------------------------------------------------

# Ensure we were invoked with a single argument .
# if len(sys.argv) != 2:
#     print(" Usage : %s < image - file >" % sys.argv[0], file=sys.stderr)
# img_path = sys.argv[1]
# print(" The filename to work on is %s ." % sys.argv[1])
img_path = "develop-001.jpg"
img = cv2.imread(img_path)

result_output = image_reader(img)

if result_output is not None:
    xpos, ypos, ang = result_output
# Output the position and bearing in the form required by the test harness .
print(" RED %0.3f %0.3f" % xpos)
print(" BLUE %0.3f %0.3f" % ypos)
print(" BEARING %0.3f" % ang)
