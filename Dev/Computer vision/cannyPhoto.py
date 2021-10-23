import cv2

path="./samples/"
cap=cv2.VideoCapture(1)
photoCount=8
while 1:
    _,frame=cap.read()
    # read a colour image from the working directory
    img = frame
    img = cv2.resize(img,(320,240))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    wide = cv2.Canny(blurred, 10, 200)
    mid = cv2.Canny(blurred, 30, 150)
    tight = cv2.Canny(blurred, 240, 250)
    # show the output Canny edge maps
    cv2.imshow("Wide Edge Map", wide)
    cv2.imshow("Mid Edge Map", mid)
    cv2.imshow("Tight Edge Map", tight)
        # display the original image
    cv2.imshow('Original Image', img)
    cv2.imshow('Blurred', blurred)

    # KEYBOARD INTERACTIONS
    
    if cv2.waitKey(1) & 0xFF == ord('s'):
        # save the image as such
        print("save")
        cv2.imwrite(path+str(photoCount)+'.jpg', img)
        cv2.imwrite(path+str(photoCount)+' blur.jpg', blurred)
        cv2.imwrite(path+str(photoCount)+' wide.jpg', wide)
        cv2.imwrite(path+str(photoCount)+' mid.jpg', mid)
        cv2.imwrite(path+str(photoCount)+' tight.jpg', tight)
        cv2.destroyAllWindows()
        photoCount+=1
    elif cv2.waitKey(1) & 0xFF==ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
