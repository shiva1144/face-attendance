import cv2 as cv
import numpy as np
import face_recognition as face_Rec
import os
import pyttsx3 as textSpeach
from datetime import datetime
import mysql.connector

engine = textSpeach.init()


# Resizing the images

def resize(img, size):
    width = int(img.shape[1] * size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv.resize(img, dimension, interpolation=cv.INTER_AREA)


path = 'Studentt'
student_Image = []
student_Name = []
my_List = os.listdir(path)

# taking the data from folder


for cl in my_List:
    curr_Image = cv.imread(f'{path}/{cl}')
    student_Image.append(curr_Image)
    student_Name.append(os.path.splitext(cl)[0])


# saving encodings from images


def find_Encoding(images):
    image_encodings = []
    for img in images:
        img = resize(img, 0.50)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encodingImg = face_Rec.face_encodings(img)[0]
        image_encodings.append(encodingImg)
    return image_encodings


# creating csv file for saving data


def Mark_attendance(name):
    cnx = mysql.connector.connect(
        user='root',
        password='qwertyuiop',
        host='localhost',
    )
    # print("cnx",cnx)

    # personName = (name)
    # print(personName)
    db = cnx.cursor()
    # print(name[0])

    countDateDiffQuery = (
        "select count(datediff(time, now())) as count_date_diff from tempdb.attendance where user_name = %s and extra "
        "= %s order by time desc limit 1")
    countDateDiffData = (name, 'extra')
    db.execute(countDateDiffQuery, countDateDiffData)
    countDateDiff = db.fetchone()

    # print("countDateDiff",countDateDiff[0])
    if countDateDiff[0] == 0:
        data = (name, 'extra')

        insertQuery = "INSERT INTO tempdb.attendance (user_name, extra) VALUES (%s, %s)"

        try:
            db.execute(insertQuery, data)
            cnx.commit()
            # print("Data inserted try")


        except:
            cnx.rollback()
            # print("Data inserted except")

        # print("Data inserted")
        print("user inserted")

    else:
        fetchUserQuery = (
            "select datediff(time, now())as date_diff from tempdb.attendance where user_name = %s and extra = %s "
            "order by time desc limit 1")
        fetchData = (name, 'extra')
        # print(fetchData)

        db.execute(fetchUserQuery, fetchData)
        userExist = db.fetchone()

        # print("select", userExist[0])

        if userExist[0] != 0:
            data = (name, 'extra')

            insertQuery = "INSERT INTO tempdb.attendance (user_name, extra) VALUES (%s, %s)"

            try:
                db.execute(insertQuery, data)
                cnx.commit()
                # print("Data inserted try")


            except:
                cnx.rollback()
                # print("Data inserted except")

            # print("Data inserted")
            print("user inserted")


        else:
            print("user already exist for today")

    # #.... fetchUserQuery = ("select datediff(time, now())as date_diff from tempdb.attendance where user_name = %s
    # and extra = %s order by time desc limit 1") fetchData = (name, 'extra') # print(fetchData)
    #
    # db.execute(fetchUserQuery,fetchData)
    # userExist = db.fetchone()
    #
    # print("select",userExist[0])
    # # .....

    # if(userExist[0] != 0 or userExist[0] == None):
    #     data = (name, 'extra')
    #
    #     insertQuery = ("INSERT INTO tempdb.attendance (user_name, extra) VALUES (%s, %s)")
    #
    #     try:
    #         db.execute(insertQuery, data)
    #         cnx.commit()
    #         # print("Data inserted try")
    #
    #
    #     except:
    #         cnx.rollback()
    #         # print("Data inserted except")
    #
    #     # print("Data inserted")
    #     print("user inserted")
    #
    #
    # else:
    #     print("user already exist for today")

    # print('connection successful')
    # # print(name)
    # insertQuery = ("INSERT INTO tempdb.attendance (name,extra) VALUES (%s, %s)")
    # personName = (name,10)
    # print(personName)
    # db.execute(insertQuery, personName)
    # # print('insertion successful')
    cnx.close()

    # with open('Attendance.csv', 'r+') as f:
    #     my_Data_List = f.readlines()
    #     name_list = []
    #     for line in my_Data_List:
    #         entry = line.split(',')
    #         name_list.append(entry[0])
    #     if name not in name_list:
    #         now = datetime.now()
    #         time_string = now.strftime('%H:%M')
    #         f.writelines(f' \n  {name}, {time_string}')
    #         statement = str('welcome to the class' + name)
    #         engine.say(statement)
    #         engine.runAndWait()


encoded_List = find_Encoding(student_Image)

vid = cv.VideoCapture(0)

# while loop for running video
while True:
    success, frame = vid.read()
    smaller_frames = cv.resize(frame, (0, 0), None, 0.25, 0.25)
    # smaller_frames = cv.cvtColor(smaller_frames, cv.COLOR_BGR2RGB)
    # finding face in the frame

    faces_In_Frame = face_Rec.face_locations(smaller_frames)
    encode_Faces_in_frame = face_Rec.face_encodings(smaller_frames, faces_In_Frame)

    # comparing faces from the frame to the trained database

    for encode_face, face_loc in zip(encode_Faces_in_frame, faces_In_Frame):
        matches = face_Rec.compare_faces(encoded_List, encode_face)
        face_Dis = face_Rec.face_distance(encoded_List, encode_face)
        print(face_Dis)
        match_Index = np.argmin(face_Dis)

        if matches[match_Index]:
            names = student_Name[match_Index].upper()
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv.rectangle(frame, (x1, y2 - 25), (x2, y2), (0, 255, 0), cv.FILLED)
            cv.putText(frame, names, (x1 + 6, y2 - 6), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            Mark_attendance(names)

    cv.imshow('video', frame)
    cv.waitKey(1)
