import numpy as np
import traceback

def read_database(dir):
    #print("Start reading data from csv file")
    dataset = []
    gesture = 0
    while True:
        path = "data/"+dir+"/gesture_"+str(gesture+1)+".csv"
        gesture = gesture + 1
        print("Open: ", path)
        try:
            data = np.loadtxt(path, delimiter=",", skiprows=2) #skip header and null point
        except:
            print("Path not found: "+path)
            break

        FrameNumber = 1
        pointlenght = 20 #maximum number of points in array
        framelenght = 50 #int(data[-1][0]) #maximum number of frames in arrat
        
        datalenght = int(len(data))
        gesturedata = np.zeros((framelenght,pointlenght, 4))
        counter = 0

        while counter < datalenght:
            arr = np.zeros((pointlenght, 4))
            
            iterator = 0

            try:
                while data[counter][0] == FrameNumber:
                    arr[iterator] = np.array([data[counter][3], data[counter][4]/1000,data[counter][5],data[counter][6]])
                    iterator += 1
                    counter += 1
            except:
                print(" ")

            try:
                gesturedata[FrameNumber - 1] = arr
            except:
                print("Frame number out of bound", counter)
                break

            FrameNumber += 1

        dataset.append(gesturedata)

    print("End of the loop")
    return dataset

dir = ["close_fist_horizontally", "close_fist_perpendicularly", "hand_to_left", "hand_to_right",
                         "hand_rotation_palm_up","hand_rotation_palm_down", "arm_to_left", "arm_to_right",
                         "hand_closer", "hand_away", "hand_up", "hand_down"]
# dir = ["close_fist_horizontally"]
#Read gestures from choosen directory
for item in dir:
    data = read_database(dir)
print(len(data))
print(data[1].shape)
