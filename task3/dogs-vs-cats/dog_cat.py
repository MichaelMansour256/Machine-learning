import numpy as np
from sklearn import svm
import os
from tqdm import tqdm
from skimage.transform import resize
from skimage.feature import hog
import cv2
path = os.getcwd()
TRAIN_DIR = f'{path}/train/train1'
TEST_DIR = f'{path}/train/test1'
Y1 = np.zeros((1000, 1))
Y2 = np.ones((1000, 1))
Y = np.concatenate((Y1, Y2))
Y = np.ravel(Y)
X = []
X_test = []
Y1_test = np.zeros((250, 1))
Y2_test = np.ones((250, 1))
Y_test = np.concatenate((Y1_test, Y2_test))
Y_test = np.ravel(Y_test)
images = []
images_test=[]
# read training data
for filename in tqdm(os.listdir(TRAIN_DIR)):
    img = cv2.imread(os.path.join(TRAIN_DIR, filename))
    if img is not None:
        images.append(img)
for img in tqdm(images):
    resized_img = resize(img, (128, 64))
    fd, hog_img = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True, multichannel=True)
    X.append(fd)
c = list(zip(X, Y))
np.random.shuffle(c)
X,Y=zip(*c)
C = 0.1  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(X, Y)  # minimize hinge oss, One vs One
lin_svc = svm.LinearSVC(C=C).fit(X, Y)  # minimize squared hinge loss, One vs All
rbf_svc = svm.SVC(kernel='rbf', gamma=0.8, C=C).fit(X, Y)
poly_svc = svm.SVC(kernel='poly', degree=5, C=C).fit(X, Y)

print("accuracy for training data")
for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    predictions = clf.predict(X)
    accuracy = np.sum(predictions == Y)
    accuracy=accuracy/2000
    print(accuracy)
print("#############################")
# read test data
for filename in tqdm(os.listdir(TEST_DIR)):
    img = cv2.imread(os.path.join(TEST_DIR, filename))
    if img is not None:
        images_test.append(img)
for img in tqdm(images_test):
    resized_img = resize(img, (128, 64))
    fd, hog_img = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True, multichannel=True)
    X_test.append(fd)
print("accuracy for testing data")
for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    predictions = clf.predict(X_test)
    accuracy = np.sum(predictions == Y_test)
    accuracy=accuracy/500
    print(accuracy)
