import cv2
import numpy as np

#this function is responsible for cartoonization
def cartoonize(img,scaleFactor=4):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray,7)
    #detecting the edges
    edges = cv2.Laplacian(img_gray,cv2.CV_8U,ksize=5)
    ret,mask = cv2.threshold(edges,100,255,cv2.THRESH_BINARY_INV)
    #thickining the lines of the sketch
    kernel = np.ones((3,3),np.uint8) 
    mask = cv2.erode(mask,kernel,iterations=1)
    #clearing the noise resulted from previous action
    mask = cv2.medianBlur(mask,3)
    #performing bilateral filteration
    iterations = 7
    size = 5
    colorNeighboorhood = 7
    spaceNeighboorhood = 5
    for i in range(iterations):
        img = cv2.bilateralFilter(img,size,colorNeighboorhood,spaceNeighboorhood)
    #placing the mask on top of the image
    dst = np.zeros(img_gray.shape)
    dst = cv2.bitwise_and(img,img,mask=mask)
    return dst

def getFaceAndCartoonize(img,faceRects):
    faces = []
    for (x,y,w,h) in faceRects:
        face = img[y:y+h,x:x+w]
        cartoonizedFace = cartoonize(face)
        img[y:y+h,x:x+w] = cartoonizedFace
    

def detectFacesAndCartoonize(img,faceCascade):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    faceRects = faceCascade.detectMultiScale(img_gray,1.3,5)
    if len(faceRects) != 0:
        getFaceAndCartoonize(img,faceRects)
        for face in faceRects:
            (x,y,w,h) = face
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    


cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
img = cv2.imread("sample.jpg")

while True:
    success , frame = cap.read()
    detectFacesAndCartoonize(frame,faceCascade)
    cv2.imshow("Cartoonized faces",frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
