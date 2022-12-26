import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from numpy import linalg as LA
import imutils
import argparse
import csv

FINAL_MARK = 'mark'
FILE_NAME = 'file_name'


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


def get_answers(img1, img2, points, is_marking_scheme, show_intermediate_results=False):
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
        img2, correspondingPoints, is_marking_scheme, show_intermediate_results)
    return answer


def calculate_score(marking_scheme, answer_script, choice_distribution):
    idd = 0
    choice_distribution = np.array(choice_distribution)
    incorrect = []
    correct = []
    for i in range(0, choice_distribution.shape[0]):
        j = 0
        for k in range(0, choice_distribution[i]):
            if (marking_scheme[idd] != answer_script[idd] and j == 0):
                incorrect.append(i+1)
                j = 1
            idd += 1
        if j == 0:
            correct.append(i+1)
    return correct, incorrect


def plot_marked_answer_sheet(marking_scheme, answer_script, template_img, bubble_coordinates, file_name='output.pdf', show_plot=False, save_plot=False):
    incorrect = []
    correct = []
    # print(np.array(pts).shape[0])
    for i in range(np.array(bubble_coordinates).shape[0]):
        if answer_script[i] == 1:
            if marking_scheme[i] != answer_script[i]:
                incorrect.append(bubble_coordinates[i])
            if marking_scheme[i] == answer_script[i]:
                correct.append(bubble_coordinates[i])

    correct = np.array(correct)
    incorrect = np.array(incorrect)
    # print(correct)
    plt.figure(figsize=(10, 10))
    plt.imshow(np.array(template_img), cmap='gray')
    if len(correct) > 0:
        plt.scatter(correct[:, 0], correct[:, 1], c='g', s=15)
    if len(incorrect) > 0:
        plt.scatter(incorrect[:, 0], incorrect[:, 1], c='r', s=15)
    plt.title(
        "Correct answers are marked green, wrong answers are marked red and unresponded are left blank")
    if show_plot:
        plt.show()
    if save_plot:
        plt.savefig(file_name)


def app():

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", help="Print detailed messages",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--markingscheme", help="Path to the bubble sheet of the marking scheme",
                        default="samples/marking_scheme.jpg")
    parser.add_argument(
        "--template", help="Path to the template of the bubble sheet", default="samples/template.jpg")
    parser.add_argument(
        "--answers", help="Path to the directory containing scanned answer scripts", default="samples/answers/")
    parser.add_argument(
        "--debug", help="Show intermediate results and debug messages", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--showmarked", help="Show the answer script marked with correct and incorrect answers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--savemarked", help="Save the answer script marked with correct and incorrect answers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--studentslist", help="Name of the file containing a list of the students' index numbers in csv format", default="samples/students_list.csv")
    parser.add_argument(
        "--output", help="Directory to save the output files such as the output.csv file containing the list of students and respective marks in csv format", default="output/")
    parser.add_argument(
        "--ignoreinputcsv", help="Ignore the input list of students given as a CSV file and use the file name instead", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    print("Running autograder...")
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    output_file = os.path.join(args.output, "output.csv")
    template_img = read_image(args.template)
    marking_scheme_img = read_image(args.markingscheme)
    student_marks = dict()
    bubble_coordinates, choice_distribution = get_coordinates_of_bubbles()
    marking_scheme = get_answers(
        template_img, marking_scheme_img, bubble_coordinates, is_marking_scheme=True, show_intermediate_results=args.debug)

    i = 0
    answer_script_files_list = sorted(glob.glob(args.answers + '*.jpg'))

    if not args.ignoreinputcsv:
        try:
            with open(args.studentslist, mode='r') as students_csv:
                csv_reader = csv.DictReader(students_csv)
                students = [row['Index No'] for row in csv_reader]
        except FileNotFoundError as e:
            print("CSV file is not available: ", e)
            students = [f.split('/')[-1] for f in answer_script_files_list]
    else:
        students = [f.split('/')[-1] for f in answer_script_files_list]

    if len(answer_script_files_list) != len(students):
        print(
            f"Mismatch in the number of entries in the provided {args.studentslist} csv file and the number of scanned scripts in the {args.answers} directory.")
        print(f"Entries in the csv file: {len(students)}")
        print(
            f"Entries in the scanned directory: {len(answer_script_files_list)}")
        sys.exit()

    for answer_script_file_path in answer_script_files_list:
        answer_script_img = read_image(answer_script_file_path)

        answer_script = get_answers(template_img, answer_script_img,
                                    bubble_coordinates, is_marking_scheme=False, show_intermediate_results=args.debug)

        correct, incorrect = calculate_score(
            marking_scheme, answer_script, choice_distribution)
        if args.verbose:
            print(
                f"Our observation for the first {len(choice_distribution)} questions are :")
            print("Correct answers are:", correct)
            print("Incorrect or Unresponded answers are:", incorrect)
        if args.showmarked or args.savemarked:
            marked_file_name = os.path.join(args.output, f"{students[i]}_output.pdf")
            plot_marked_answer_sheet(
                marking_scheme, answer_script, template_img, bubble_coordinates, file_name=marked_file_name, show_plot=args.showmarked, save_plot=args.savemarked)

        print(
            f"Result for {students[i]}: {len(correct)}/90, Incorrect: {len(incorrect)}/90")

        student_marks[students[i]] = {FINAL_MARK: len(
            correct), FILE_NAME: answer_script_file_path}
        i += 1

    # write the autograded output to the csv file
    with open(output_file, 'w') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['Index No', 'Autograded Final Mark', 'Answer Script'])
        for key, value in student_marks.items():
            writer.writerow([key, value[FINAL_MARK], value[FILE_NAME]])

    print(
        f"Autograding is complete. Output has been saved in {args.output}.")


if __name__ == "__main__":
    app()
