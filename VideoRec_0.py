#!/usr/bin/env python
# coding: utf-8

# In[2]:


import face_recognition
import cv2


# In[7]:


video_capture = cv2.VideoCapture("/dev/video0")


# In[5]:


i = 0
#plt.figure()
while True:
    i += 1
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    for top, right, bottom, left in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.imshow('CV1', frame)
        # plt.imshow(frame, cmap = 'gray', interpolation = 'bicubic') 
        # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        # plt.show()

    if cv2.waitKey(1) & 0xFF == ord('q'):
       break


# In[5]:


video_capture.release()
cv2.destroyAllWindows()


# In[4]:





# In[ ]:




