#!/usr/bin/env python
# coding: utf-8

# In[1]:


import face_recognition
import cv2


# In[4]:





# In[27]:


image_1 = face_recognition.load_image_file("/home/serg/Документи/PythonScripts/ComputerVision/Image2Rec_Son.jpg")
image_2 = face_recognition.load_image_file("/home/serg/Документи/PythonScripts/ComputerVision/Image2Rec_Mom.jpg")
face_encoding_1 = face_recognition.face_encodings(image_1)[0]
face_encoding_2 = face_recognition.face_encodings(image_2)[0]
known_faces = [
face_encoding_1,
face_encoding_2,
]


# In[31]:


video_capture = cv2.VideoCapture("/dev/video0")


# In[32]:


face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    ret, frame = video_capture.read()
    frame_number += 1
    
    # Quit when the input video file ends
    if not ret:
       break
    
    rgb_frame = frame[:, :, ::-1]
    
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    
    for face_encoding in face_encodings:
       # See if the face is a match for the known face(s)
       match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

       name = None
       if match[0]:
           name = "Partitor"
       elif match[1]:
           name = "Best Mom of the World"

       face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue
        
        cv2.rectangle(frame, (left-6, top-6), (right+6, bottom+35), (0, 0, 255), 2)#, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom-6), font, 0.5, (255, 255, 255), 1)

        # Write the resulting image to the output video file
        # print("Writing frame {} / {}".format(frame_number, length))
        cv2.imshow('CV2', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break


# In[33]:


video_capture.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




