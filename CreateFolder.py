import os


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  OK  ---")
        return True
    else:
        print("---  There is this folder!  ---")
        return False


if __name__ == "__main__":
    Path = r"D:/UTA Real-Life Drowsiness Dataset AU Preprocessing/Group{0}/"
    for i in range(1, 6):
        nowPath = Path.format(i)
        folderList = os.listdir(nowPath)
        for folder in folderList:
            samplePath = nowPath + folder
            mkdir(samplePath + "/corrcoef/")
            # mkdir(samplePath + "/highMapping/")
