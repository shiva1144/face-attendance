import cv2
import numpy as npy
import face_recognition as face_rec


# function
def resize(img, size):
    width = int(img.shape[1] * size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)


# img declaration
Ankit = face_rec.load_image_file('Students\Ankit.jpg')
Ankit = cv2.cvtColor(Ankit, cv2.COLOR_BGR2RGB)
Ankit = resize(Ankit, 0.50)
Ashish_test = face_rec.load_image_file('Students\Ashish_Shukla.jpg')
Ashish_test = resize(Ashish_test, 0.50)
Ashish_test = cv2.cvtColor(Ashish_test, cv2.COLOR_BGR2RGB)

Aditya_test = face_rec.load_image_file('Students\Ashish_Shukla.jpg')
Aditya_test = resize(Ashish_test, 0.50)
Aditya_test = cv2.cvtColor(Ashish_test, cv2.COLOR_BGR2RGB)



# finding face location

faceLocation_Ankit = face_rec.face_locations(Ankit)[0]
encode_Ankit = face_rec.face_encodings(Ankit)[0]
cv2.rectangle(Ankit, (faceLocation_Ankit[3], faceLocation_Ankit[0]), (faceLocation_Ankit[1], faceLocation_Ankit[2]),
              (255, 0, 255), 3)

faceLocation_ashishTest = face_rec.face_locations(Ashish_test)[0]
encode_AshishTest = face_rec.face_encodings(Ashish_test)[0]
cv2.rectangle(Ashish_test, (faceLocation_Ankit[3], faceLocation_Ankit[0]),
              (faceLocation_Ankit[1], faceLocation_Ankit[2]), (255, 0, 255), 3)

results = face_rec.compare_faces([encode_Ankit], encode_AshishTest)
print(results)
cv2.putText(Ashish_test, f'{results}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('main_img', Ankit)
cv2.imshow('test_img', Ashish_test)
cv2.waitKey(0)
cv2.destroyAllWindows()
