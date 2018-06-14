import os
import numpy as np
import cv2

train_data_num = 60000  # The number of training figures
test_data_num = 10000  # The number of testing figures
fig_w = 45  # width of each figure
new_size = (28, 28)

def read_origin_data():  # read original mnist data
    print("Reading original mnist data...")
    # file path
    train_data_path = os.path.join("mnist", "mnist_train", "mnist_train_data")
    train_label_path = os.path.join("mnist", "mnist_train", "mnist_train_label")
    test_data_path = os.path.join("mnist", "mnist_test", "mnist_test_data")
    test_label_path = os.path.join("mnist", "mnist_test", "mnist_test_label")
    # read data
    train_data = np.fromfile(train_data_path, dtype=np.uint8)
    train_label = np.fromfile(train_label_path, dtype=np.uint8)
    test_data = np.fromfile(test_data_path, dtype=np.uint8)
    test_label = np.fromfile(test_label_path, dtype=np.uint8)
    # reshape
    train_data = train_data.reshape(train_data_num, fig_w*fig_w)
    test_data = test_data.reshape(test_data_num, fig_w*fig_w)
    return train_data, train_label, test_data, test_label

def read_data(stage=1):  # read processed data
    # stage == 0: original data
    # stage == 1: remove noise area
    # stage == 2: remove redundant background
    # stage == 3: binary
    # stage == "HOG": HOG
    if stage == 0:  # not processed
        return read_origin_data()
    if stage == "HOG":
        hog = True
        stage = 2
    else:
        hog = False
    print("Reading processed mnist data...")
    # file path
    processed_train_data_path = os.path.join("mnist", "data", "processed_mnist_train_data_"+str(stage))
    train_label_path = os.path.join("mnist", "mnist_train", "mnist_train_label")
    processed_test_data_path = os.path.join("mnist", "data", "processed_mnist_test_data_"+str(stage))
    test_label_path = os.path.join("mnist", "mnist_test", "mnist_test_label")
    # read data
    train_data = np.fromfile(processed_train_data_path, dtype=np.uint8)
    train_label = np.fromfile(train_label_path, dtype=np.uint8)
    test_data = np.fromfile(processed_test_data_path, dtype=np.uint8)
    test_label = np.fromfile(test_label_path, dtype=np.uint8)
    # reshape
    if stage == 1:
        size = (fig_w, fig_w)
    elif stage >= 2:
        size = new_size
    train_data = train_data.reshape(train_data_num, size[0] * size[1])
    test_data = test_data.reshape(test_data_num, size[0] * size[1])
    if hog:
        train_data = hog_feature(train_data)
        test_data = hog_feature(test_data)
    return train_data, train_label, test_data, test_label

def process_data(stage=1):  # processing data
    print("Stage:", stage)
    train_data, train_label, test_data, test_label = read_data(stage-1)  # read data from the previous stage
    # reshape
    if stage <= 2:
        size = (fig_w, fig_w)
    elif stage > 2:
        size = new_size
    train_data = train_data.reshape(train_data_num, size[0], size[1])
    test_data = test_data.reshape(test_data_num, size[0], size[1])
    # processing
    if stage == 2:
        print("Finding the best size...")
        max_h, max_w = 0, 0
        train_data_rect, test_data_rect = list(), list()
        print("In training data: ")
        for i in range(train_data_num):
            x, y, h, w = min_rect(train_data[i])
            max_h, max_w = max(max_h, h), max(max_w, w)
            train_data_rect.append((x, y, h, w))
            show_process(i, train_data_num)
        print("\nIn testing data: ")
        for i in range(test_data_num):
            x, y, h, w = min_rect(test_data[i])
            max_h, max_w = max(max_h, h), max(max_w, w)
            test_data_rect.append((x, y, h, w))
            show_process(i, test_data_num)
        print("\nNew size: h=%d, w=%d" %(max_h, max_w))
        new_train_data = np.zeros((train_data_num, max_h, max_w), dtype=np.uint8)
        new_test_data = np.zeros((test_data_num, max_h, max_w), dtype=np.uint8)
        print("Resize training data...")
        for i in range(train_data_num):
            x, y, h, w = train_data_rect[i]
            new_x, new_y = (max_h - h) // 2, (max_w - w) // 2
            new_train_data[i][new_x:new_x+h,new_y:new_y+w] = train_data[i][x:x+h,y:y+w]
            show_process(i, train_data_num)
        print("\nResize testing data...")
        for i in range(test_data_num):
            x, y, h, w = test_data_rect[i]
            new_x, new_y = (max_h - h) // 2, (max_w - w) // 2
            new_test_data[i][new_x:new_x+h,new_y:new_y+w] = test_data[i][x:x+h,y:y+w]
            show_process(i, test_data_num)
        train_data, test_data = new_train_data, new_test_data
        size = (max_h, max_w)
    else:  # stage 1 & 3
        print("Process training data...")
        for i in range(train_data_num):
            train_data[i] = process_img(train_data[i], stage)
            show_process(i, train_data_num)
        print()
        print("Process testing data...")
        for i in range(test_data_num):
            test_data[i] = process_img(test_data[i], stage)
            show_process(i, test_data_num)
        print()
    # reshape
    if stage == 1:
        size = (fig_w, fig_w)
    elif stage > 2:
        size = new_size
    train_data = train_data.reshape(train_data_num * size[0] * size[1])
    test_data = test_data.reshape(test_data_num * size[0] * size[1])
    # save
    processed_train_data_path = os.path.join("mnist", "data", "processed_mnist_train_data_"+str(stage))
    processed_test_data_path = os.path.join("mnist", "data", "processed_mnist_test_data_"+str(stage))
    train_data.tofile(processed_train_data_path)
    test_data.tofile(processed_test_data_path)

def show_process(now, total):
    if now % 100 == 0:
        print("\r%d/%d" %(now, total), end="")

def flood_fill(img, region, x0, y0, val):  # fill areas with val
    if region[x0,y0] > 0:
        return 0
    area = 1
    region[x0,y0] = val
    direction = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    direction = [(-1,0),(0,-1),(0,1),(1,0)]
    for d in direction:
        x, y = x0 + d[0], y0 + d[1]
        if x < 0 or x >= fig_w or y < 0 or y >= fig_w:
            continue
        area += flood_fill(img, region, x, y, val)
    return area

def remove_noise_region(img):  # remove small areas from the image
    area = list()
    region = np.zeros((fig_w,fig_w), dtype=np.uint8)
    region[img==0] = 255  # background
    val = 1
    for i in range(fig_w):
        for j in range(fig_w):
            a = flood_fill(img, region, i, j, val)
            if a > 0:
                area.append(a)
                val += 1
    max_area = np.argmax(area) + 1
    img[region!=max_area] = 0  # black bg
    #img[region==max_area] = 255  # white digit
    return img

def min_rect(img):  # the min rect of the digit
    top, bottom, left, right = 0, fig_w - 1, 0, fig_w - 1
    while sum(img[top,:]) == 0:
        top += 1
    while sum(img[bottom,:]) == 0:
        bottom -= 1
    while sum(img[:,left]) == 0:
        left += 1
    while sum(img[:,right]) == 0:
        right -= 1
    x, y = top, left
    h, w = bottom - top + 1, right - left + 1
    return x, y, h, w

def process_img(img, stage):
    '''#test
    img = remove_noise_region(img)
    cv2.imwrite("test1.png",img)
    x, y, h, w = min_rect(img)
    print(x, y, h, w)
    img = img[x:x+h,y:y+w]
    cv2.imwrite("test2.png",img)
    val, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite("test3.png",img)
    exit(0)'''
    if stage == 1:  # remove noise region
        img = remove_noise_region(img)
    elif stage == 3: # binary
        val, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return img

def hog_feature(data):
    features = list()
    winSize = new_size
    blockSize = (14,14)
    blockStride = (7,7) 
    cellSize = (7,7)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    print("Calculating HOG features...")
    total = data.shape[0]
    for i in range(total):
        img = data[i].reshape(new_size[0], new_size[1])
        hog_feature = hog.compute(img)
        features.append(hog_feature)
        show_process(i, total)
    print()
    features = np.array(features)
    return features.reshape(total, features.shape[1])

def main():
    process_data(stage=2)

if __name__=="__main__":
    main()
