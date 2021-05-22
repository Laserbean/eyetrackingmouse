import os
import sys
import glob

import dlib


trainfolder = "/pic"

options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = True
options.C = 5
# Tell the code how many CPU cores your computer has for the fastest training.
options.num_threads = 4
options.be_verbose = True

training_xml_path = "/pic/a.xml"

#testing_xml_path = "/pic/testing.xml"
dlib.train_simple_object_detector(training_xml_path, "detector.svm", options)
print("done training")


# Now that we have a face detector we can test it.  The first statement tests
# it on the training data.  It will print(the precision, recall, and then)
# average precision.
print("")  # Print blank line to create gap from previous output
print("Training accuracy: {}".format(
    dlib.test_simple_object_detector(training_xml_path, "detector.svm")))
# However, to get an idea if it really worked without overfitting we need to
# run it on images it wasn't trained on.  The next line does this.  Happily, we
# see that the object detector works perfectly on the testing images.
print("Testing accuracy: {}".format(
    dlib.test_simple_object_detector(testing_xml_path, "detector.svm")))


