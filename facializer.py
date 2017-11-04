"""
Author: Will Irwin
Date: 10/8/2017
Inputs:

    Training pictures in specific folder for a person
    Picture(s) in specific folder from camera to test
Outputs:
    true for if image contains person
    false else
"""


# coding: utf-8

# Face Recognition with OpenCV

# To detect faces, I will use the code from my previous article on [face detection](https://www.superdatascience.com/opencv-face-detection/). So if you have not read it, I encourage you to do so to understand how face detection works and its Python coding.

# ### Import Required Modules

# Before starting the actual coding we need to import the required modules for coding. So let's import them first.
#
# - **cv2:** is _OpenCV_ module for Python which we will use for face detection and face recognition.
# - **os:** We will use this Python module to read our training directories and file names.
# - **numpy:** We will use this module to convert Python lists to numpy arrays as OpenCV face recognizers accept numpy arrays.

# In[1]:

#import OpenCV module
import cv2
#import os module for reading training data directories and paths
import os
#import numpy to convert python lists to numpy arrays as
#it is needed by OpenCV face recognizers
import numpy as np
from random import randint
import time

# ### Training Data

# The more images used in training the better. Normally a lot of images are used for training a face recognizer so that it can learn different looks of the same person, for example with glasses, without glasses, laughing, sad, happy, crying, with beard, without beard etc. To keep our tutorial simple we are going to use only 12 images for each person.
#
# So our training data consists of total 2 persons with 12 images of each person. All training data is inside _`training-data`_ folder. _`training-data`_ folder contains one folder for each person and **each folder is named with format `sLabel (e.g. s1, s2)` where label is actually the integer label assigned to that person**. For example folder named s1 means that this folder contains images for person 1. The directory structure tree for training data is as follows:
#
# ```
# training-data
# |-------------- s1
# |               |-- 1.jpg
# |               |-- ...
# |               |-- 12.jpg
# |-------------- s2
# |               |-- 1.jpg
# |               |-- ...
# |               |-- 12.jpg
# ```
#
# The _`test-data`_ folder contains images that we will use to test our face recognizer after it has been successfully trained.

# As OpenCV face recognizer accepts labels as integers so we need to define a mapping between integer labels and persons actual names so below I am defining a mapping of persons integer labels and their respective names.
#
# **Note:** As we have not assigned `label 0` to any person so **the mapping for label 0 is empty**.

# In[2]:

#there is no label 0 in our training data so subject name for index/label 0 is empty
#subjects = ["UNKNOWN", "Dagan Martinez","Will Irwin","Jordan Faison", "Logan Micher"]


# ### Prepare training data

# You may be wondering why data preparation, right? Well, OpenCV face recognizer accepts data in a specific format. It accepts two vectors, one vector is of faces of all the persons and the second vector is of integer labels for each face so that when processing a face the face recognizer knows which person that particular face belongs too.
#
# For example, if we had 2 persons and 2 images for each person.
#
# ```
# PERSON-1    PERSON-2
#
# img1        img1
# img2        img2
# ```
#
# Then the prepare data step will produce following face and label vectors.
#
# ```
# FACES                        LABELS
#
# person1_img1_face              1
# person1_img2_face              1
# person2_img1_face              2
# person2_img2_face              2
# ```
#
#
# Preparing data step can be further divided into following sub-steps.
#
# 1. Read all the folder names of subjects/persons provided in training data folder. So for example, in this tutorial we have folder names: `s1, s2`.
# 2. For each subject, extract label number. **Do you remember that our folders have a special naming convention?** Folder names follow the format `sLabel` where `Label` is an integer representing the label we have assigned to that subject. So for example, folder name `s1` means that the subject has label 1, s2 means subject label is 2 and so on. The label extracted in this step is assigned to each face detected in the next step.
# 3. Read all the images of the subject, detect face from each image.
# 4. Add each face to faces vector with corresponding subject label (extracted in above step) added to labels vector.
#
# **[There should be a visualization for above steps here]**

# Did you read my last article on [face detection](https://www.superdatascience.com/opencv-face-detection/)? No? Then you better do so right now because to detect faces, I am going to use the code from my previous article on [face detection](https://www.superdatascience.com/opencv-face-detection/). So if you have not read it, I encourage you to do so to understand how face detection works and its coding. Below is the same code.

# In[3]:

#function to detect face using OpenCV
def detect_face(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]

    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]


# I am using OpenCV's **LBP face detector**. On _line 4_, I convert the image to grayscale because most operations in OpenCV are performed in gray scale, then on _line 8_ I load LBP face detector using `cv2.CascadeClassifier` class. After that on _line 12_ I use `cv2.CascadeClassifier` class' `detectMultiScale` method to detect all the faces in the image. on _line 20_, from detected faces I only pick the first face because in one image there will be only one face (under the assumption that there will be only one prominent face). As faces returned by `detectMultiScale` method are actually rectangles (x, y, width, height) and not actual faces images so we have to extract face image area from the main image. So on _line 23_ I extract face area from gray image and return both the face image area and face rectangle.
#
# Now you have got a face detector and you know the 4 steps to prepare the data, so are you ready to code the prepare data step? Yes? So let's do it.

# In[4]:

#this function will read all persons' training images, detect face from each image
#and will return two lists of exactly same size, one list
# of faces and another list of labels for each face
def prepare_training_data(data_folder_path):

    #------STEP-1--------
    #get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)

    #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []

    #let's go through each directory and read images within it
    for dir_name in dirs:

        #our subject directories start with letter 's' so
        #ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue;

        #------STEP-2--------
        #extract label number of subject from dir_name
        #format of dir name = slabel
        #, so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))

        #build path of directory containin images for current subject subject
        #sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name

        #get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)

        #------STEP-3--------
        #go through each image name, read image,
        #detect face and add face to list of faces
        for image_name in subject_images_names:

            #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;

            #build image path
            #sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            #read image
            image = cv2.imread(image_path)

            #display an image window to show the image
            #cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            #cv2.waitKey(5)

            #detect face
            try:
                face, rect = detect_face(image)
            except:
                print("Fail")
                continue

            #------STEP-4--------
            #for the purpose of this tutorial
            #we will ignore faces that are not detected
            if face is not None:
                #add face to list of faces
                faces.append(face)
                #add label for this face
                labels.append(label)
            else:
                print("NFD:"+image_path)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels

def validate(data_folder_path):

    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    confidence_threshold = 0.0
    for dir_name in dirs:

        if not dir_name.startswith("s"):
            continue;

        label = int(dir_name.replace("s", ""))

        subject_dir_path = data_folder_path + "/" + dir_name

        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:

            #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;

            #build image path
            #sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            #read image
            image = cv2.imread(image_path)

            #detect face
            face, rect = detect_face(image)

            if face is not None:
                #add face to list of faces
                faces.append(face)
                #add label for this face
                labels.append(label)
            else:
                pass

    #list to hold labels for all subjects
    thresholds = []
    #sets the threshold of a correct guess
    for face in faces:
        guess = face_recognizer.predict(face)
        #print(guess)
        thresholds.append(guess[1])
    #print(thresholds)
    confidence_threshold = (sum(thresholds)/len(thresholds)) + 3

    return confidence_threshold

def predict(test_img, confidence_threshold):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img)

    if face is not None:
        pass
    else:
        return 0

    #predict the image using our face recognizer
    label, confidence = face_recognizer.predict(face)
    label_text = "aknown"
    try:
        label_text = subjects[label]
    except:
        pass
        
    return label_text
    """
    #get name of respective label returned by face recognizer
    if label[1] > confidence_threshold:
        label_text = subjects[label[0]]
    else:
        label_text = subjects[0]

    return label_text
    """

# Now that we have the prediction function well defined, next step is to actually call this function on our test images and display those test images to see if our face recognizer correctly recognized them. So let's do it. This is what we have been waiting for.

# In[10]:


def main(capture, confidence_threshold):
    #load test images
    test_img = cv2.imread(capture)
    if test_img is None:
        print("FUCK FUCK FUCK FUCK")
    #perform a prediction
    #predicted_img = predict(test_img)
    prediction = None
    try:
        prediction = predict(test_img, confidence_threshold)
    except:
        print("well shit")
        prediction = predict(test_img, confidence_threshold)
        return False
    #print("Prediction complete")
    if prediction == subjects[2]:
        return True
    else:
        return False
    #display all images
    #cv2.imshow(subjects[2], cv2.resize(predicted_img, (400, 500)))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.waitKey(1)
    #cv2.destroyAllWindows()
    #return 0

test_location = "test-data/"
training_location = "training-data"
validation_location = "validating-data"
subjects = ["0", "relaxed", "stressed", "-1"]

#let's first prepare our training data
#data will be in two lists of same size
#one list will contain all the faces
#and other list will contain respective labels for each face
print("Preparing data...")
print("")
faces, labels = prepare_training_data(training_location)
print("Data prepared")

#print total faces and labels
print("Total faces: "+ str(len(faces)))
print("Total labels: "+ str(len(labels)))
print(labels)
print("")

#create our LBPH face recognizer
def _createFaceRecognizer():
	return cv2.face.LBPHFaceRecognizer_create()
	#return cv2.face.createLBPHFaceRecognizer()
face_recognizer = _createFaceRecognizer()

#train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))

#test with separate pictures known correct
confidence_threshold = validate(validation_location)


def isCorrectFace(capture):
	return main(capture, confidence_threshold)

if __name__=="__main__":
    print("TESTING")
    for name in os.listdir("training-data/s1"):
        if name[0]  == ".":
            continue
        fulldir = "training-data/s1/"+name

        print(isCorrectFace(fulldir))
    

# Use like so: from facializer import isCorrectFace
"""
while True:
    time.sleep(5)
    if len(os.listdir(test_location)) != 1:
        #need to mess with this to make sure it looks at and deletes the oldest file
        #and leaves newest 1 to prevent errors
        capture = test_location + os.listdir(test_location)[len(os.listdir(test_location)) -1]
        #print(capture)
        main(capture, confidence_threshold)
        os.remove(capture)
    else:
        pass
"""
