# OMR based MCQ Autograder

MCQAutoGrader implements an Optical Mark Recognition based autograding tool for paper based MCQ questions. The project is mainly designed to be used for CS1033 Programming Fundamentals course offered by the Dept. of Computer Science and Engineering at University of Moratuwa, Sri Lanka.

MCQAutoGrader is designed to take a list of scanned answer scripts (bubble sheets) as inputs along with the marking scheme and the template of the bubble sheet. The template of the bubble sheet is available [here](https://docs.google.com/spreadsheets/d/1oUphoxSrNf3qI7_DLRZII-zN9sUES-WGTxp9o_Qo21Q/edit?usp=sharing).

The project uses [poetry](https://python-poetry.org/docs/) for dependency management and packaging. To set up the dependencies run the following command after cloning/downloading the repository.

`poetry install`

To find more information about the tool execute the following command after you have set up the project:

`python3 mcqautograder/autograder.py --help`

Typical use of the tool require the template, marking scheme, directory containing scanned answer scripts, a list of students as a csv file to be provided as follows. Refer the [samples](/samples/) directory.

`python3 mcqautograder/autograder.py --template samples/template.jpg --markingscheme samples/marking_scheme.jpg --answers samples/answers/ --studentslist samples/students_list.csv`

MCQAutoGrader also contains a multiversion_autograder tool to grade exam papers with multiple versions. Typical use of this tool would require templates and marking schemes for each version, directory containing scanned answer scripts, and a list of students and the corresponding exam paper version given to the student as a csv file to be provided as follows. Refer the [samples](/samples/) directory.

`python3 mcqautograder/multiversion_autograder.py --templates samples/templates/ --markingschemes samples/marking_schemes/ --answers samples/answers/ --studentslist samples/students_list.csv --numversions 2`

Please note that here the templates and corresponding marking schemes should be renamed to just the version number (eg: 1.jpg, 2.jpg, ...).

If you have any questions, please contact: [gayashan@cse.mrt.ac.lk](mailto:gayashan@cse.mrt.ac.lk)

This project is inspired and based on the following work. Special thanks should go to these authors for their contributions:

[1] Chidrewar, Vaibhav, Junwei Yang, and Donguk Moon. "Mobile based auto grading of answersheets." (2014).

[2] Jain, Salay and Sharma, Harsh, "OMR-Auto-Grading-System", https://github.com/salay-jain/OMR-Auto-Grading-System