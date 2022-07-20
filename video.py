#Installing necessary packages
!apt update
!apt install -y cmake
!pip install dlib
!git clone https://github.com/nicolasmetallo/eameo-faceswap-generator
cd eameo-faceswap-generator

#Importing necessary packages
import cv2
import os
import torch
import numpy as np
import math import matplotlib.pyplot as plt
%matplotlib inline
import dlib
import faceBlendCommon as fbc
from google.colab.patches import cv2_imshow
from sklearn import svm
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#preprocessing
def ref_preprocess(img):
    face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_alt.xml')
    detected_faces = face_cascade.detectMultiScale(img)
    count=1
    x,y=0,0
    fc=[]
    L=0
    for (column, row, width, height) in detected_faces:
        L=width
        xc=height//2

        yc=width//2
        face_region=img[row:row+height,column:column+width]
        fc=face_region
        cv2.imwrite(str(count)+'faces.jpg', face_region)
        count+=1
    temp_fc=face_region
    #print(xc,yc)
    detector=dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    points2 = fbc.getLandmarks(detector, predictor,face_region)
    vs=pow((points2[27][1]-points2[28][1]),2)
    vs=vs/pow((points2[29][1]-points2[28][1]),2)
    hs=pow((points2[0][0]-yc),2)
    hs=hs/pow((points2[16][0]-yc),2)

    reg=[]
    reg.append(list(range(68)))
    reg.append(list(range(17)))
    reg.append(list(range(17,22)))
    reg.append(list(range(22,27)))
    reg.append(list(range(36,42)))
    reg.append(list(range(42,48)))
    reg.append(list(range(48,61)))
    reg.append([61,62,63,65,66,67])
    reg[-2].append(64)
    desc=[]
    desc1=[]
    d0=[]
    d1=[]
    d2=[]
    d3=[]
    d4=[]
    d5=[]
    d6=[]
    d7=[]
    for k in range(8):
        r=reg[k]
        des=[]
        des1=[]
        ln=len(r)
        xsum=0
        ysum=0
        for i in r:
            xsum+=points2[i][0]
            ysum+=points2[i][1]

        xcn=xsum/ln
        ycn=ysum/ln
        d22=0
        for i in r:
            d22=(pow((xcn-points2[i][0]),2)+pow((ycn-points2[i][1]),2))/pow(L,2)
            if points2[i][0]<=xcn:
                d22=d22*hs*hs
            if points2[i][1]<=ycn:
                d22=d22*vs*vs
            des1.append(d22)
            if d22==0:
                d22=0.0001
            s=math.log(d22)
            des.append(s)
        desc.append(des)
        eval(f"d{k}.append({des})")
        desc1.append(des1)
    return desc #returning descriptors

    #Dataset preprocessing
    data_path = '/content/drive/MyDrive/Dataset miniproject/ck+/ck+./ck/ck+'
    data_dir_list = os.listdir(data_path)
    img_data_list=[]
    d0=[]
    d1=[]
    d2=[]
    d3=[]
    d4=[]
    d5=[]
    d6=[]
    d7=[]
    for dataset in data_dir_list:
        img_list=os.listdir(data_path+'/'+ dataset)
        print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
        for img in img_list:
            input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
            #input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            print(dataset+'/'+img)
            input_img_prep=ref_preprocess(input_img)
            img_data_list.append(input_img_prep)
            for i in range(8):
                eval(f"d{i}.append({input_img_prep[i]})")

#labelling Dataset
num_classes = 7
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')
labels[0:44]=0 #
labels[45:103]=1 #
labels[104:127]=2 #
labels[128:186]=3 #
labels[187:214]=4 #
labels[215:273]=5 #
labels[274:323]=6 #
names = ['anger','disgust','fear','happy','sad','surprise','contempt']

def getLabel(id):
    return ['anger','disgust','fear','happy','sad','surprise','contempt'][id]

#8 svm classifiers for 8 descriptors
clf0 = svm.SVC(decision_function_shape='ovo',probability=True)
Y = np_utils.to_categorical(labels, num_classes)
x,y = shuffle(d0_data,labels, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)
#X_train = X_train.reshape(275, 136, 68, 1)
x_test=X_test
clf0.fit(X_train,y_train)
y_pred=clf0.predict(X_test) acc_svc=clf0.score(x_test,y_test)
print("accuracy 0",acc_svc) y_pred0=clf0.predict_proba([X_test[0]])
print(y_pred0)

clf1 = svm.SVC(decision_function_shape='ovo',probability=True)
Y = np_utils.to_categorical(labels, num_classes)
x,y = shuffle(d1_data,labels, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)
#X_train = X_train.reshape(275, 136, 68, 1)
x_test=X_test clf1.fit(X_train,y_train)
y_pred=clf1.predict(X_test)
acc_svc=clf1.score(x_test,y_test)
print("accuracy 1",acc_svc)
y_pred1=clf1.predict_proba([X_test[0]])
print(y_pred1)

clf2 = svm.SVC(decision_function_shape='ovo',probability=True)
Y = np_utils.to_categorical(labels, num_classes)
x,y = shuffle(d2_data,labels, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)
#X_train = X_train.reshape(275, 136, 68, 1)
x_test=X_test clf2.fit(X_train,y_train)
y_pred=clf2.predict(X_test)
acc_svc=clf2.score(x_test,y_test)
print("accuracy 2",acc_svc)
y_pred2=clf2.predict_proba([X_test[0]])
print(y_pred2)

clf3 = svm.SVC(decision_function_shape='ovo',probability=True)
Y = np_utils.to_categorical(labels, num_classes)
x,y = shuffle(d3_data,labels, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)
#X_train = X_train.reshape(275, 136, 68, 1)
x_test=X_test
clf3.fit(X_train,y_train)
y_pred=clf3.predict(X_test)
acc_svc=clf3.score(x_test,y_test)
print("accuracy 3",acc_svc)
y_pred3=clf3.predict_proba([X_test[0]])
print(y_pred3)

clf4 = svm.SVC(decision_function_shape='ovo',probability=True)
Y = np_utils.to_categorical(labels, num_classes)
x,y = shuffle(d4_data,labels, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)
#X_train = X_train.reshape(275, 136, 68, 1)
x_test=X_test clf4.fit(X_train,y_train)
y_pred=clf4.predict(X_test)
acc_svc=clf4.score(x_test,y_test)
print("accuracy 4",acc_svc)
y_pred4=clf4.predict_proba([X_test[0]])
print(y_pred4)

clf5 = svm.SVC(decision_function_shape='ovo',probability=True)
Y = np_utils.to_categorical(labels, num_classes)
x,y = shuffle(d5_data,labels, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)
#X_train = X_train.reshape(275, 136, 68, 1)
x_test=X_test clf5.fit(X_train,y_train)
y_pred=clf5.predict(X_test)
acc_svc=clf5.score(x_test,y_test)
print("accuracy 5",acc_svc)
y_pred5=clf5.predict_proba([X_test[0]])
print(y_pred5)

clf6 = svm.SVC(decision_function_shape='ovo',probability=True)
Y = np_utils.to_categorical(labels, num_classes)
x,y = shuffle(d6_data,labels, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)
#X_train = X_train.reshape(275, 136, 68, 1)
x_test=X_test clf6.fit(X_train,y_train)
y_pred=clf6.predict(X_test)
acc_svc=clf6.score(x_test,y_test)
print("accuracy 6",acc_svc)
y_pred6=clf6.predict_proba([X_test[0]])
print(y_pred6)

clf7 = svm.SVC(decision_function_shape='ovo',probability=True)
Y = np_utils.to_categorical(labels, num_classes)
x,y = shuffle(d7_data,labels, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)
#X_train = X_train.reshape(275, 136, 68, 1)
x_test=X_test clf7.fit(X_train,y_train)
y_pred=clf7.predict(X_test)
acc_svc=clf7.score(x_test,y_test)
print("accuracy 7",acc_svc)
y_pred7=clf7.predict_proba([X_test[0]])
print(y_pred7)

#Getting predicted probabilities from 8 svm classifiers and passing the as input to ensemble REFER svm classifier
data_path = '/content/drive/MyDrive/Dataset miniproject/ck+/ck+./ck/ck+'
data_dir_list = os.listdir(data_path)
Refer_des=[]
for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
        #input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        print(dataset+'/'+img) input_img_prep=ref_preprocess(input_img)
        #img_data_list.append(input_img_prep)
        y_pred0=clf0.predict_proba([input_img_prep[0]])
        y_pred1=clf1.predict_proba([input_img_prep[1]])
        y_pred2=clf2.predict_proba([input_img_prep[2]])
        y_pred3=clf3.predict_proba([input_img_prep[3]])
        y_pred4=clf4.predict_proba([input_img_prep[4]])
        y_pred5=clf5.predict_proba([input_img_prep[5]])
        y_pred6=clf6.predict_proba([input_img_prep[6]])
        y_pred7=clf7.predict_proba([input_img_prep[7]])
        l=[y_pred0,y_pred1,y_pred2,y_pred3,y_pred4,y_pred5,y_pred6,y_pred7]
        Refer_des.append(l)
#Final REFER classifier
Ref_clf = svm.SVC(decision_function_shape='ovo',probability=True)
Y = np_utils.to_categorical(labels, num_classes)
x,y = shuffle(rf,labels, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)
#X_train = X_train.reshape(275, 136, 68, 1)
x_test=X_test Ref_clf.fit(X_train,y_train)
y_pred=Ref_clf.predict(X_test)
acc_svc=Ref_clf.score(x_test,y_test)
print("accuracy Refer",acc_svc)
y_predr=Ref_clf.predict_proba([X_test[0]])
print(y_predr,Ref_clf.predict([X_test[0]]))
