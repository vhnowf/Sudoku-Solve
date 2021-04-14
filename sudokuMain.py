from google.colab.patches import cv2_imshow
import cv2
import numpy as np
from tensorflow.keras.models import load_model


#### Đọc mô hình CNN
def intializePredectionModel():
    model = load_model('Resources/myModel.h5')
    return model


#### 1 - Tiền sử lý ảnh
def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Chuyển hình ảnh thành trắng đen
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # Làm mờ ảnh
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)  # Nhị phân hóa ảnh
    return imgThreshold


#### 2 - Sắp xếp lại các điểm cho Warp Perspective
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


#### 3 - Tìm đường viền lớn nhất
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area


#### 4 - Chia bức ảnh thành 81 bức ảnh nhỏ hơn
def splitBoxes(img):
    rows = np.vsplit(img,9)
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes


#### 5 - Phán đoán trên tất cả các ảnh
def getPredection(boxes,model):
    result = []
    for image in boxes:
        ## Chuẩn bị ảnh
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] -4]
        img = cv2.resize(img, (28, 28))
        img = img / 255
        img = img.reshape(1, 28, 28, 1)
        ## Phán đoán
        predictions = model.predict(img)
        classIndex = model.predict_classes(img)
        probabilityValue = np.amax(predictions)
        ## Lưu kết quả
        if probabilityValue > 0.8:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result


#### 6 -  Hiện thị đáp án trên ảnh
def displayNumbers(img,numbers,color = (0,255,0)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range (0,9):
        for y in range (0,9):
            if numbers[(y*9)+x] != 0 :
                 cv2.putText(img, str(numbers[(y*9)+x]),
                               (x*secW+int(secW/2)-10, int((y+0.8)*secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, color, 2, cv2.LINE_AA)
    return img


#### 6 - Vẽ thêm các đường kẻ ô 
def drawGrid(img):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for i in range (0,9):
        pt1 = (0,secH*i)
        pt2 = (img.shape[1],secH*i)
        pt3 = (secW * i, 0)
        pt4 = (secW*i,img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0),2)
        cv2.line(img, pt3, pt4, (255, 255, 0),2)
    return img
import warnings
warnings.filterwarnings('ignore')
import sudukoSolver
########################################################################
print('Welcome to Sudoku Solver')
print('Pleases enter Sudoku image name!')
imagename = input()
pathImage = "Resources/"+imagename
heightImg = 450
widthImg = 450
model = intializePredectionModel()  # tải mô hình CNN
########################################################################


#### 1. Xủ lý ảnh
img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImg, heightImg))  # Chỉnh lại cỡ ảnh thành hình vuông
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # Tạo một ảnh rỗng
imgThreshold = preProcess(img) 

# #### 2. Tìm toàn bộ các đường viền bao quanh
imgContours = img.copy() 
imgBigContour = img.copy() 
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
cv2.drawContours(imgContours, contours, -1, (0, 255, 255), 3) # Vẽ mầu lên toàn bộ đường viền bao

#### 3. Tìm đường viền bao ngoài lớn
biggest, maxArea = biggestContour(contours) 
# print(biggest)
if biggest.size != 0:
    biggest = reorder(biggest)
    # print(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25) # Chấm 4 góc lên đường viền bao lớn nhất
    pts1 = np.float32(biggest) # Chuẩn bị các điểm cho WARP
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # Chuẩn bị các điểm cho WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg)) # Biến đổi phối cảnh
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
  
    #### 4. Chia nhỏ ảnh và tìm các số có sẵn
    imgSolvedDigits = imgBlank.copy()
    boxes = splitBoxes(imgWarpColored)
    # print(len(boxes))
    numbers = getPredection(boxes,model) #Dự đoán các số có sẵn
    # print(numbers)
    imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(0, 255, 255)) # Hiện thị các số có sẵn
    imgDetectedDigits = drawGrid(imgDetectedDigits)
    numbers = np.asarray(numbers)
    posArray = np.where(numbers > 0, 0, 1)
    # print(posArray)


    #### 5. Tìm các số còn thiếu
    board = np.array_split(numbers,9) # Tạo thành 9 hàng
    # print(board)
    try:
        sudukoSolver.solve(board) #Giải bài toán
    except:
        pass
    # print(board)
    flatList = [] # Đưa 9 hàng về 1 hàng
    for sublist in board:
        for item in sublist:
            flatList.append(item)
    solvedNumbers =flatList*posArray #Tạo thành danh sách chỉ có các số là đáp án không có các số có sẵn
    imgSolvedDigits= displayNumbers(imgSolvedDigits,solvedNumbers,color=(0, 0, 255))
    

    # #### 6. Phủ kết quả lên ảnh gốc
    pts2 = np.float32(biggest) # Set 
    pts1 =  np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) 
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  
    imgInvWarpColored = img.copy()
    imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
    inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
    imgSolvedDigits = drawGrid(imgSolvedDigits)
    cv2_imshow(img)
    print('Solution')
    cv2_imshow(inv_perspective)
else:
    print("No Sudoku Found")

