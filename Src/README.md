# Eye Tracking Mouse with Web Cam

## Introduction

> Ever wanted to control the computer with your eyes? Whether it's cause you can't use your hands, or if you're multitasking, or you're just plain lazy, this project is just for you (Not yet. It's still in development)

> This code uses pytorch's Convolutional Neural Network and dlib facial landmark recognition. It uses Auto hotkey (AHK) to move the mouse. 


## Installation 

> No installation required. 
> To run the software, run the myRun.sh file using bash or .\


## Usage

> "a" to the toggle on the "AI mode" which allows eye tracking to directly control the mouse input. 
> "t" toggles test mode. This will compare the gaze estimation with the actual cursor position. Make sure you look at the cursor before starting the test mode. 


## Training your own data
> "n" toggles the data capture mode. It will move the mouse cursor in a grid pattern and take a few pictures of the eye. This is only used for the data training. 
> "m" toggles the continuous capture mode. This will continuously capture pictures of the eye as it follows the mouse cursor which can be moved to anywhere on the screen. 
> "c" just takes a photo of the eye and remembers the location of the mouse cursor. 

> All pictures should be moved into a new directory. 
> To train with the new data the following line, 253, should be edited 
>> totaldata = dataLoad(["directory1", "directory2", "directory3"])
>>> Note that there can be as many directorys with test data as needed. 
> If test data is present edit line 261 like how 253 was. 
>> testdata = dataLoad(["testDirectory1", "testDirectory2"])

> Note: All png files are named with the following format: 
>> <x_coordinate>.<y_coordinate>.<image_number>.png






