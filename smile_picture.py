import cv2

video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")

while True:
    success, img = video.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)
    cnt = 1
    key_pressed = cv2.waitKey(1)

    for x, y, w, h in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 3)
        smiles = smile_cascade.detectMultiScale(gray_img, 1.8, 15)
        for i, j, k, l in smiles:
            img = cv2.rectangle(img, (i, j), (i + k, j + l), (100, 100, 100), 5)
            print("Image " + str(cnt) + "Saved")

            path = r'/Users/rohan/IdeaProjects/facial_recognition/img' + str(cnt) + '.jpg'
            cv2.imwrite(path, img)
            cnt += 1
            if cnt >= 2:
                break

    cv2.imshow('live video', img)
    if key_pressed & 0xFF == ord('q'):
        break


video.release()
cv2.destroyAllWindows()
