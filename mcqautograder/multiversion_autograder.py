
import argparse
import csv
import glob
import sys
import os

from autograder import calculate_score, get_answers, get_coordinates_of_bubbles, plot_marked_answer_sheet, read_image, FILE_NAME, FINAL_MARK

VERSION = 'version'

def app():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", help="Print detailed messages",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--markingschemes", help="Path to the bubble sheets of the marking schemes for each version",
                        default="samples/marking_schemes/")
    parser.add_argument(
        "--templates", help="Path to the templates of the bubble sheets for each version", default="samples/templates/")
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
        "--numversions", help="Number of versions", default=2, type=int)

    args = parser.parse_args()

    print("Running multi version autograder...")
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    output_file = os.path.join(args.output, "output.csv")
    template_imgs = {}
    marking_scheme_imgs = {}
    marking_schemes = {}
    student_marks = {}
    students = {}

    # read templates
    for i in range(1, args.numversions + 1):
        try:
            template_imgs[i] = read_image(os.path.join(args.templates, f"{i}.jpg"))
        except FileNotFoundError as e:
            print(f"Template file is not found: ", e)
            sys.exit()

    # read marking schemes
    for i in range(1, args.numversions + 1):
        try:
            marking_scheme_imgs[i] = read_image(os.path.join(args.markingschemes, f"{i}.jpg"))
        except FileNotFoundError as e:
            print(f"Marking scheme file is not found: ", e)
            sys.exit()

    bubble_coordinates, choice_distribution = get_coordinates_of_bubbles()


    for i in range(1, args.numversions + 1):
        marking_schemes[i] = get_answers(template_imgs[i], marking_scheme_imgs[i], bubble_coordinates, is_marking_scheme=True, show_intermediate_results=args.debug)

    try:
        with open(args.studentslist, mode='r') as students_csv:
            csv_reader = csv.DictReader(students_csv)
            for row in csv_reader:
                students[row['Index No']] = row['version']
    except FileNotFoundError as e:
        print("CSV file is not available: ", e)
        sys.exit()

    answer_script_files_list = sorted(glob.glob(args.answers + '*.jpg'))

    if len(answer_script_files_list) != len(students):
        print(
            f"Mismatch in the number of entries in the provided {args.studentslist} csv file and the number of scanned scripts in the {args.answers} directory.")
        print(f"Entries in the csv file: {len(students)}")
        print(
            f"Entries in the scanned directory: {len(answer_script_files_list)}")
        sys.exit()

    for answer_script_file_path, student in zip(answer_script_files_list, students.keys()):
        answer_script_img = read_image(answer_script_file_path)

        exam_paper_version = int(students[student])
        answer_script = get_answers(template_imgs[exam_paper_version], answer_script_img,
                                    bubble_coordinates, is_marking_scheme=False, show_intermediate_results=args.debug)

        correct, incorrect = calculate_score(marking_schemes[exam_paper_version], answer_script, choice_distribution)
        if args.verbose:
            print(
                f"Marks for {len(choice_distribution)} questions: ")
            print("Correct answers are:", correct)
            print("Incorrect or Unresponded answers are:", incorrect)
        if args.showmarked or args.savemarked:
            marked_file_name = os.path.join(args.output, f"{student}_output.png")
            plot_marked_answer_sheet(
                marking_schemes[exam_paper_version], answer_script, template_imgs[exam_paper_version], bubble_coordinates, file_name=marked_file_name, show_plot=args.showmarked, save_plot=args.savemarked)

        print(
            f"Result for {student}: {len(correct)}/90, Incorrect: {len(incorrect)}/90, Version: {exam_paper_version}")

        student_marks[student] = {FINAL_MARK: len(correct), VERSION: exam_paper_version, FILE_NAME: answer_script_file_path}

    # write the autograded output to the csv file
    with open(output_file, 'w') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['Index No', 'Autograded Final Mark', 'Version', 'Answer Script'])
        for key, value in student_marks.items():
            writer.writerow([key, value[FINAL_MARK], value[VERSION], value[FILE_NAME]])

    print(
        f"Autograding is complete. Output has been saved in {args.output}.")


if __name__ == "__main__":
    app()
