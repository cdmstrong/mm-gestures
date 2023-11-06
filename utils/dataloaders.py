import torch
from torch.utils.data import Dataset, DataLoader
from utils.common import gestureIdx
import numpy as np
import traceback

class mm_DataSet(Dataset):

    def __init__(self):
        self.data = []
        self.labels = []
        self.get_data()

    def get_data(self):
        dir = ["close_fist_horizontally", "close_fist_perpendicularly", "hand_to_left", "hand_to_right",
                         "hand_rotation_palm_up","hand_rotation_palm_down", "arm_to_left", "arm_to_right",
                         "hand_closer", "hand_away", "hand_up", "hand_down"]
        dir = ["close_fist_horizontally"]
        #Read gestures from choosen directory
        for item in dir:
            data = read_database(item)
            for data_item in data:
                self.data.append(data_item)
                self.labels.append(gestureIdx[item].value)
        print(len(self.data))
        print(self.labels[30])
        # print(data[1].shape)
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def read_database(dir):
    #print("Start reading data from csv file")
    dataset = []
    gesture = 0
    while True:
        print(dir)
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
    print(len(dataset))
    return dataset

# dir = ["close_fist_horizontally", "close_fist_perpendicularly", "hand_to_left", "hand_to_right",
#                          "hand_rotation_palm_up","hand_rotation_palm_down", "arm_to_left", "arm_to_right",
#                          "hand_closer", "hand_away", "hand_up", "hand_down"]
# dir = ["close_fist_horizontally"]
# #Read gestures from choosen directory
# data = read_database(dir[0])
# print(data[1].shape)

if __name__ == "__main__":
    # print(gestureIdx.hand_away.value)
    # mm_DataSet()
    dataset = mm_DataSet()
    # cached_dataset = torch.utils.data.dataset.DatasetCacher(dataset)

    train_size = int(0.8 * len(dataset))  # 80% 数据用于训练
    val_size = len(dataset) - train_size  # 剩余20% 数据用于验证

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    # 创建数据加载器并使用缓存
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    for x, label in train_loader:
        x = x.view(x.shape[0], x.shape[1], -1)
        print(x.shape)
