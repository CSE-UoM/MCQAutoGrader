import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from numpy import linalg as LA
import imutils
import argparse


def read_image(path):
    img = Image.open(path).convert('L')
    img.resize((1000, 1000))
    return img


def get_homography(img1, img2):
    orig_image = np.array(img1)
    skewed_image = np.array(img2)
    # using surf for feature matching
    try:
        surf = cv2.xfeatures2d.SURF_create(500)
    except Exception:
        surf = cv2.SIFT_create()
    # Finding the matching features between the two images
    kp1, des1 = surf.detectAndCompute(orig_image, None)
    kp2, des2 = surf.detectAndCompute(skewed_image, None)
    # Using the FLANN detector to remove the outliers
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    # Setting the min match count for the match count of labels
    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H


def get_coordinates_of_bubbles():
    x = 126  # starting x coordinate of the first bubble
    y = 145  # starting y coordinate of the first bubble
    coordinates = []
    for i in range(3):
        for j in range(30):
            for k in range(5):
                a = [x+(k*34), y+(j*28)]
                coordinates.append(a)
        x += 219
    # The number of choices of each question in a sequential manner
    choice_distribution = [5 for i in range(90)]
    return coordinates, choice_distribution


def get_corresponding_points(points, H):
    points = np.array(points)
    x = points.shape[0]
    # Appending one for the homography to work for the points matching
    point = np.hstack((points, np.ones((x, 1))))
    point = point.T
    # print(H.shape)
    # print(point.shape)
    correspondingPoints = np.matmul(H, point)
    correspondingPoints = correspondingPoints.T
    for i in range(0, x):
        correspondingPoints[i][0] = correspondingPoints[i][0] / \
            correspondingPoints[i][2]
        correspondingPoints[i][1] = correspondingPoints[i][1] / \
            correspondingPoints[i][2]
    # Returning the corresponding points for the 2nd image related to 1st image
    return correspondingPoints


def check_neighbours_pixels(img, points, is_marking_scheme=True, show_intermediate_results=False):
    img = np.array(img)
    points = np.array(points)
    points = points.astype('int')
    # Thresholding the image using threshold
    binaryImg = (img < 200).astype(np.uint8)
    # To improve our accuracy we do opening to succesfully white circles
    kernel = np.ones((5, 5), np.uint8)
    binaryImg = cv2.morphologyEx(binaryImg, cv2.MORPH_OPEN, kernel)

    if show_intermediate_results:
        plt.figure(figsize=(15, 15))
        plt.imshow(binaryImg, cmap='gray')
        if is_marking_scheme:
            plt.title("Points matching the marking scheme")
        else:
            plt.title("Points matching the student's answer sheet")
        plt.scatter(points[:, 0], points[:, 1], s=10)
        plt.show()

    x = points.shape[0]
    n = 5
    # finding number of average white pixels around all the points in the inverted image
    answers = np.zeros(x)
    for i in range(0, x):
        ans = 0
        for j in range(points[i, 0]-n, points[i, 0]+n):
            for k in range(points[i, 1]-n, points[i, 1]+n):
                # plt.scatter(j,k)
                if (binaryImg[k][j]):
                    ans += 1
        answers[i] = ans
    answers = answers > 0
    return answers.astype('int')


def get_answers(img1, img2, points, is_marking_scheme):
    # Find homography Matrix
    homography = get_homography(img1, img2)
    # Find related points in the two image
    correspondingPoints = get_corresponding_points(points, homography)
    # plt.figure(figsize=(10,10))
    # plt.imshow(np.array(img2),cmap='gray')
    # plt.scatter(correspondingPoints[:,0],correspondingPoints[:,1])
    # plt.show()

    # Check neighbouring pixels and get whether option is marked or not
    answer = check_neighbours_pixels(
        img2, correspondingPoints, is_marking_scheme=is_marking_scheme)
    return answer


def calculate_score(o, y, distribution):
    idd = 0
    distribution = np.array(distribution)
    wrong = []
    right = []
    for i in range(0, distribution.shape[0]):
        j = 0
        for k in range(0, distribution[i]):
            if (o[idd] != y[idd] and j == 0):
                wrong.append(i+1)
                j = 1
            idd += 1
        if j == 0:
            right.append(i+1)
    return right, wrong


def plot_marked_answer_sheet(o, t, img, pts):
    wrong = []
    correct = []
    # print(np.array(pts).shape[0])
    for i in range(np.array(pts).shape[0]):
        if t[i] == 1:
            if o[i] != t[i]:
                wrong.append(pts[i])
            if o[i] == t[i]:
                correct.append(pts[i])

    correct = np.array(correct)
    wrong = np.array(wrong)
    # print(correct)
    plt.figure(figsize=(15, 15))
    plt.imshow(np.array(img), cmap='gray')
    if len(correct) > 0:
        plt.scatter(correct[:, 0], correct[:, 1], c='g', s=15)
    if len(wrong) > 0:
        plt.scatter(wrong[:, 0], wrong[:, 1], c='r', s=15)
    plt.title(
        "Correct answers are marked green, wrong answers are marked red and unresponded are left blank")
    plt.show()


# parser = argparse.ArgumentParser()
# parser.add_argument("--inp1", help="Path to template image")
# parser.add_argument("--inp2", help="Path to ideal answer key")
# parser.add_argument("--inp3", help="Path of to be checked")
# args = parser.parse_args()


template_img = read_image(
    "/home/gayashan/projects/cse/2022/MCQAutoGrader/samples/template.jpg")
marking_scheme_img = read_image(
    "/home/gayashan/projects/cse/2022/MCQAutoGrader/samples/marking_scheme.jpg")
answer_script_img = read_image(
    "/home/gayashan/Downloads/samples/SKM_558e22122315350_0002.jpg")

print("Running...")


bubble_coordinates, choice_distribution = get_coordinates_of_bubbles()

marking_scheme = get_answers(
    template_img, marking_scheme_img, bubble_coordinates, is_marking_scheme=True)
answer_script = get_answers(template_img, answer_script_img,
                            bubble_coordinates, is_marking_scheme=False)

correct, wrong = calculate_score(
    marking_scheme, answer_script, choice_distribution)
print(
    f"Our observation for the first {len(choice_distribution)} questions are :")
print("Correct answers are:", correct)
print("Incorrect or Unresponded answers are:", wrong)
plot_marked_answer_sheet(marking_scheme, answer_script,
                         template_img, bubble_coordinates)
print(f"Result: {len(correct)}/90, Wrong: {len(wrong)}/90")