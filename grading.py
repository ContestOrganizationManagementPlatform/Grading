import cv2
import numpy as np
import math
import pytesseract
from dataclasses import dataclass
from PIL import Image

# Stores all information relating to a potential answer box in an accessible class
@dataclass
class Answer:
    rect: list
    # question number. Will be updated later on in the code
    number: int = 0

    def __post_init__(self):
        self.xmin = min([self.rect[0][0][0], self.rect[1][0][0], self.rect[2][0][0], self.rect[3][0][0]])
        self.xmax = max([self.rect[0][0][0], self.rect[1][0][0], self.rect[2][0][0], self.rect[3][0][0]])

        self.ymin = min([self.rect[0][0][1], self.rect[1][0][1], self.rect[2][0][1], self.rect[3][0][1]])
        self.ymax = max([self.rect[0][0][1], self.rect[1][0][1], self.rect[2][0][1], self.rect[3][0][1]])
        self.yrange = self.ymax-self.ymin
        
        self.area = (self.xmax - self.xmin) * (self.ymax - self.ymin)

        # self.image is the cropped image which will be passed to graders
        self.image = image[self.ymin-10:self.ymax+10, self.xmin-10:self.xmax+10]

        # self.image_nobox is the cropped image, minus the boxes around the answer
        if self.ymax - self.ymin > 10 and self.xmax - self.xmin > 10:
            self.image_nobox = image[self.ymin+5:self.ymax-5, self.xmin+5:self.xmax-5]

    # Calculate whether an answer box overlaps with a list of other answer boxes
    def overlap(self, rects):
        for r in rects:
            if not (r.xmax < self.xmin or r.xmin > self.xmax):
                if not (r.ymax < self.ymin or r.ymin > self.ymax):
                    return True
        return False


# Finds edges in an image
def find_edges(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve contour detection
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Perform edge detection using the Canny edge detector
    edges = cv2.Canny(blurred_image, 50, 150)
    return edges

# Returns all answer boxes
def find_answer_regions(edges):
    # Find contours in the edges image
    contours,_ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    potential_answers = []

    for i, contour in enumerate(contours):
        # Approximate the contour to a polygon
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the polygon has four vertices (a rectangle)
        if len(approx) == 4:
            potential_answers.append(approx)

    # Create rectangle objects
    for i in range(len(potential_answers)):
        potential_answers[i] = Answer(rect = potential_answers[i])

    # Sort potential answers by size
    potential_answers = sorted(potential_answers, key=lambda x: x.area, reverse = True)
    answers = []

    for i in range(len(potential_answers)):
        # Check if rectangle is within the central box AND if rectangle isn't equal to central box
        if potential_answers[i].overlap([potential_answers[0]]):
            if potential_answers[i].area < 0.9*potential_answers[0].area:
                # Make sure answer does not overlap with previous answers
                if not potential_answers[i].overlap(answers): 
                    answers.append(potential_answers[i])
    
    return answers


def assign_question_number(answers):
    for i, answer in enumerate(answers):
        if answer.area > 100:
            edges = find_edges(answer.image_nobox)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=0, minLineLength=answer.yrange/2, maxLineGap=10)

            # Iterate over the detected lines to find dividing line between question number and response
            x1 = x2 = math.inf
            if str(lines) == "None":
                lines = []
            for line in lines:
                if line[0][0] + line[0][2] < x1 + x2:
                    slope = (line[0][3] - line[0][1]) / (line[0][2] - line[0][0]) if (line[0][2] - line[0][0]) != 0 else float('inf')
                    if abs(slope) > 1:
                        x1 = line[0][0]
                        x2 = line[0][2]
            if x1 != math.inf:
                answer.number = int(pytesseract.image_to_string(answer.image_nobox[:, :min([x1,x2])-5], config='--psm 10'))
                cv2.imshow(str(answer.number), answer.image_nobox[:, min([x1,x2])+10:])
                print(str(answer.number) + " [" + pytesseract.image_to_string(answer.image_nobox[:, min([x1,x2])+10:], config='--psm 10') + "]\n\n\n")
                answer.image_nobox = answer.image_nobox[:, min([x1,x2])+10:]

def store_info(answers):
    for answer in answers:
        image = Image.fromarray(answer.image)
        image.save("res/box/" + str(answer.number) + ".png")
        image = Image.fromarray(answer.image_nobox)
        image.save("res/nobox/" + str(answer.number) + ".png")

# Load an image from file
image = cv2.imread('samples/s16.png')
answers = find_answer_regions(find_edges(image))
assign_question_number(answers)
store_info(answers)

# Show each answer box
for answer in answers:
    cv2.imshow(f'Answer {answer.number}', answer.image)


# Display the result
cv2.imshow('Entry Submitted', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
