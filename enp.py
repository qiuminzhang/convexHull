import cv2
import numpy as np
import sys

"""
Assume each function is an internal node.
"""


def read_image():
    if (len(sys.argv)) < 2:
        file_path = "sample.jpg"
    else:
        file_path = sys.argv[1]

        # read image
    src = cv2.imread(file_path, 1)
    return src


"""
Convert image to binary image
1. convert image to gray scale
2. blur image
3. binarize the image
"""


def cvt_color(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def blur_image(gray):
    blur = cv2.blur(gray, (3, 3))
    return blur


def binarize_image(blur):
    ret, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
    return ret, thresh


# TBD: if this node should return all the three variables
def find_contours(thresh):
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, \
                                                cv2.CHAIN_APPROX_SIMPLE)
    return im2, contours, hierarchy


def find_convex_hull(contours):
    # create hull array for convexHull points
    hull = []

    # calculate points for each contour
    for i in range(len(contours)):
        hull.append(cv2.convexHull(contours[i], False))

    return hull


def draw_convex_hull(hull, thresh, contours, hierarchy):
    # create an empty black image
    drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)

    # draw contours and hull points
    for i in range(len(contours)):
        color_contours = (0, 255, 0)  # color for contours
        color = (255, 255, 255)  # color for convex hull
        # draw contours
        cv2.drawContours(drawing, contours, i, color_contours, 2, 8, hierarchy)
        # draw convex hull
        cv2.drawContours(drawing, hull, i, color, 2, 8)

    return drawing


def show_image(win_name, image):
    cv2.imshow(win_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():

    image = read_image()

    show_image("source", image)

    # convert image to grayscale
    grey = cvt_color(image)

    # blur image
    blur = blur_image(grey)

    # binarize image
    _, thresh = binarize_image(blur)

    # find contours
    _, contours, hierarchy = find_contours(thresh)

    # find convex hull and points
    hull = find_convex_hull(contours)

    # draw convex hull
    drawing = draw_convex_hull(hull, thresh, contours, hierarchy)

    show_image("convex hull", drawing)


if __name__ == "__main__":
    main()
