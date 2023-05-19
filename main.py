import cv2

capture = cv2.VideoCapture(0)
face = cv2.CascadeClassifier('default_frontal_face.xml')

while True:
    success, img = capture.read()
    faces = face.detectMultiScale(img, 1.2, 4)
    for (x, y, w, h) in faces:
        image1 = img[y:y + h, x:x + w]
        gaussian_blurr = cv2.GaussianBlur(image1, (91, 91), 0)

        img[y:y + h, x:x + w] = gaussian_blurr

    if faces == ():
        cv2.putText(img, 'No Face Found!', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
    cv2.imshow('Face Blur', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

capture.release()

cv2.destroyAllWindows()
