
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')


# In[2]:


datadir = "./data/"
wordsFile = datadir + "words.txt"


# In[3]:


def loadWords():
    columns=['filename', 'word', 'greylvl']

    with open(wordsFile, 'r') as words:
        rowsList = []
        for line in words:

            # if comment
            if line[0] == "#":
                continue

            data = line.split()
            data = [data[0], data[8], data[2]]
            row = dict( (colName, data[i]) for i, colName in enumerate(columns))

            rowsList.append(row)

    df = pd.DataFrame(rowsList, columns=columns)
    return df


# In[4]:


df = loadWords()
print(df.head())
print(df.shape)


# In[5]:


def preprocessImg(filename):
    # Read and load
    filename = filename.split("-")
    path = "/".join([datadir + "words", filename[0], "-".join(filename[:2]), "-".join(filename)+'.png'])
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

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
    pattern[0:newSize[1], 0:newSize[0]] = newImg

    del img, newImg

    return pattern


# In[6]:


def batchGenerator(batchSize=512):
    start = 0
    stop  = df.shape[0]

    while start + batchSize < stop:
        pathes = df.loc[start:start+batchSize-1, 'filename']
        imgs = [preprocessImg(path) for path in pathes]
        batch = np.stack(imgs, axis=0)
        print("return batch %d:%d" % (start, start+batchSize))
        yield batch

        del pathes, imgs, batch
        start += batchSize


# In[ ]:


for batch in batchGenerator():
    print(batch.shape)
