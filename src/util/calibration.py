import cv2 as cv2
import numpy as np
import threading
import queue
import time

Const_Cali_Window_name = 'canvas'
# Const_Display_X = 1680
# Const_Display_Y = 1050

Const_Display_X = 400
Const_Display_Y = 300

Const_Cali_Num_X = 3
Const_Cali_Num_Y = 3
Const_Cali_Radius = 30
Const_Cali_Resize_Radius = 7


Const_Cali_Move_Duration = 0.5          # 캘리브레이션 원 이동 속도
Const_Cali_Caputure_Duration = 0.8      # 캘리브레이션 원 줄어드는 속도

Const_Cali_Margin_X = 50
Const_Cali_Margin_Y = 50

Const_Cali_Cross_Size = 16


Cali_Center_Points = []

sequence = queue.Queue()




