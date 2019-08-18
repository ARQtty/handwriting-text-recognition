
# coding: utf-8

# In[3]:


import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os.path import getsize

datadir = "./data/"
wordsFile = datadir + "words.txt"


# # Load dataframe with metainf

# In[4]:


def loadWords():
    columns=['filename', 'word', 'greylvl']

    with open(wordsFile, 'r') as words:
        rowsList = []
        for line in words:

            # if comment
            if line[0] == "#":
                continue

            data = line.split()
            word = data[8]
            # dataset contains 1 word length 53
            if len(word) > 32:
                continue

            data = [data[0], data[8], data[2]]
            row = dict( (colName, data[i]) for i, colName in enumerate(columns))

            rowsList.append(row)

    df = pd.DataFrame(rowsList, columns=columns)
    return df


# In[5]:



# # Function for prepare each image
# Resizes and pastes into (32, 128) pattern. This shape is equal to receptive field of nn

# In[6]:



def preprocessImg(filename):
    # Read and load
    filename = filename.split("-")
    path = "/".join([datadir + "words", filename[0], "-".join(filename[:2]), "-".join(filename)+'.png'])

    if not (getsize(path)):
        print("Corrupted file "+path)
        return np.zeros((32, 128))
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    except:
        print("Problem with loading file "+path)
        return np.zeros((32, 128))


    # Resize
    (targetW, targetH) = (128, 32)
    (imgH, imgW) = img.shape

    fy = targetH / imgH
    fx = targetW / imgW
    f = min(fx, fy)

    newSize = (int(np.trunc(imgW*f)), int(np.trunc(imgH*f)))
    newImg = cv2.resize(img, newSize)


    # Fill to NN pattern
    pattern = np.ones((32, 128)) * 255
    pattern[0:newSize[1], 0:newSize[0]] =  newImg#№np.trunc(newImg * 255)

    return pattern


# In[7]:


# # Image data generator

# In[9]:

def loadCharList():
    df = loadWords()
    charList = set()

    for word in list(df.loc[:, 'word']):
        for c in list(word):
            charList.add(c)

    return "".join(sorted(charList))


def batchGenerator(batchSize=512, mode='train'):
    df = loadWords()

    start = 0
    # 95% for train
    stop = int(df.shape[0] * 0.95)
    num = 0

    if (mode == 'test'):
        # 5% for validation
        start = int(df.shape[0]*0.95)
        stop = df.shape[0]

    while start + batchSize < stop:
        num += 1
        pathes = df.loc[start:start+batchSize, 'filename']
        imgs = [preprocessImg(path) for path in pathes]
        gtTexts = list(df.loc[start:start+batchSize, 'word'])
        batch = (np.stack(imgs, axis=0), gtTexts)
        start += batchSize

        yield batch


# In[11]:


if __name__ == '__main__':
    df = loadWords()
    print(df.head())
    print("Dataframe shape:", df.shape)

    plt.imshow(preprocessImg("a01-000u-00-02"), cmap='gray')

    for batch in batchGenerator(batchSize=8192, mode='train'):
        print(batch.shape)
