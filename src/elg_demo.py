#!/usr/bin/env python3
"""Main script for gaze direction inference from webcam feed."""
import argparse
import os
import queue
import threading
import time

import coloredlogs
import cv2 as cv
import cv2 as cv2
import numpy as np
import tensorflow as tf

# from util.calibration import *
from datasources import Video, Webcam
from models import ELG
import util.gaze

start_condition = False
is_detect = False

###################################### Start Cali #############################################
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




def start_cali():

    # 큐에 캘리브레이션 순서 인덱스를 찾례로 넣는다
    for i in range(0, Const_Cali_Num_X * 1):
        sequence.put_nowait(i)

    img = init_canvas()
    background = img.copy()
    init_cali()

    draw_circle(img, Cali_Center_Points[0], Const_Cali_Radius)
    draw_cross(img, Cali_Center_Points[0])

    for point in Cali_Center_Points:
        draw_circle(img, point, Const_Cali_Radius)
        draw_cross(img, point)
    # 큐의 순서대로 캘리브레이션 시작
    index = sequence.get_nowait()
    resize_figure(img, Cali_Center_Points[index], Const_Cali_Radius, Const_Cali_Caputure_Duration, background)

    cv2.waitKey(0)
    close_window(Const_Cali_Window_name)
    return


def move_figure(img, start_point, end_point, current_point, duration, background, count = 0):

    global is_detect
    while( is_detect == False):
        continue

    img = background.copy()

    Const_Unit_Time = 60        # 0.01초마다 실행
    to_move_x = (end_point[0] - start_point[0])
    to_move_y = (end_point[1] - start_point[1])

    move_once_x = to_move_x / (duration * Const_Unit_Time)
    move_once_y = to_move_y / (duration * Const_Unit_Time)

    updated_current_point = (current_point[0] + move_once_x , current_point[1] + move_once_y)

    draw_circle(img, current_point, Const_Cali_Radius)
    draw_cross(img, current_point)

    display_canvas(Const_Cali_Window_name,img)
    count = count +1 

    if (count == (duration * Const_Unit_Time)):
        resize_figure(img, end_point, Const_Cali_Radius, Const_Cali_Caputure_Duration, background )
        return

    # threading.Timer(1 / Const_Unit_Time, move_figure, [img, start_point, end_point, updated_current_point, duration, background, count]).start()
    th = threading.Timer(1 / Const_Unit_Time, move_figure, [img, start_point, end_point, updated_current_point, Const_Cali_Move_Duration, background, count])
    th.daemon = True
    th.start()


def resize_figure(img, point, current_radius, duration, background, count = 0):




    img = background.copy()

    Const_Unit_Time = 60        # 0.01초마다 실행

    to_resize_radius = Const_Cali_Radius - Const_Cali_Resize_Radius
    resize_once_radius = to_resize_radius    / (duration * Const_Unit_Time)

    updated_current_radius = current_radius - resize_once_radius
    draw_circle(img, point, updated_current_radius)
    draw_cross(img, point)

    display_canvas(Const_Cali_Window_name,img)
    count = count + 1

    global is_detect
    while( is_detect == False):
        continue

    # 캘리브레이션 순간!
    if(count == (duration * Const_Unit_Time)):
        # 다음 캘리브레이션 경로가 없다면 창 종료
        if (sequence.empty() == True):
            return
        ##########################################
        # to-do : 눈의 좌표 저장 

        ##########################################

        # 큐에 다음 캘리브레이션 포인트가 있다면 원을 이동하여 캘리브레이션 작업
        index = sequence.get_nowait()
        move_figure(img, point, Cali_Center_Points[index], point, Const_Cali_Move_Duration, background )
        return

    th = threading.Timer(1 / Const_Unit_Time, resize_figure, [img, point, updated_current_radius, duration, background, count])
    th.daemon = True
    th.start()

def init_cali():
    cali_unit_distance_x = (Const_Display_X - Const_Cali_Margin_X * 2) / (Const_Cali_Num_X - 1)
    cali_unit_distance_y = (Const_Display_Y - Const_Cali_Margin_Y * 2) / (Const_Cali_Num_Y - 1)


    for y in range(0, Const_Cali_Num_Y):
        for x in range(0, Const_Cali_Num_X):
            Cali_Center_Points.append(((int)(Const_Cali_Margin_X + cali_unit_distance_x * x) 
                                     , (int)(Const_Cali_Margin_Y + cali_unit_distance_y * y) ))



def init_canvas():
    img = np.zeros((Const_Display_Y, Const_Display_X, 3), np.uint8)
    return img

def draw_circle(img , point, radius):
    img = cv2.circle(img, ((int)(point[0]),(int)(point[1])), (int)(radius), (0,0,255), -1)

def draw_cross(img , point):
    half_size = (int)(Const_Cali_Cross_Size / 2)
    img = cv2.line(img, ((int)(point[0] - half_size), (int)(point[1])) , ((int)(point[0] + half_size), (int)(point[1])) , (255,255,255),1)
    img = cv2.line(img, ((int)(point[0]) , (int)(point[1] - half_size)) , ((int)(point[0]) , (int)(point[1] + half_size)) , (255,255,255),1)


def display_canvas(canvas_name , img):
    # Display On Canvas
    # cv2.namedWindow(canvas_name, cv2.WINDOW_NORMAL);  # auto resized
    # cv2.setWindowProperty(canvas_name, cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow(canvas_name, img)
    None

def close_window(canvas_name):
    # cv2.destroyWindow(canvas_name)
    cv2.destroyAllWindows()


###################################### End Cali #############################################3


if __name__ == '__main__':


    # Set global log level
    parser = argparse.ArgumentParser(description='Demonstration of landmarks localization.')
    parser.add_argument('-v', type=str, help='logging level', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    parser.add_argument('--from_video', type=str, help='Use this video path instead of webcam')
    parser.add_argument('--record_video', type=str, help='Output path of video of demonstration.')
    parser.add_argument('--fullscreen', action='store_true')
    parser.add_argument('--headless', action='store_true')

    parser.add_argument('--fps', type=int, default=60, help='Desired sampling rate of webcam')
    parser.add_argument('--camera_id', type=int, default=0, help='ID of webcam to use')

    args = parser.parse_args()
    coloredlogs.install(
        datefmt='%d/%m %H:%M',
        fmt='%(asctime)s %(levelname)s %(message)s',
        level=args.v.upper(),
    )

    # Check if GPU is available
    from tensorflow.python.client import device_lib
    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    gpu_available = False


    try:
        gpus = [d for d in device_lib.list_local_devices(session_config)
                if d.device_type == 'GPU']
        gpu_available = len(gpus) > 0

    except:
        pass

    # Initialize Tensorflow session
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Session(config=session_config) as session:

        # Declare some parameters
        batch_size = 2

        # Define webcam stream data source
        # Change data_format='NHWC' if not using CUDA
        if args.from_video:
            assert os.path.isfile(args.from_video)
            data_source = Video(args.from_video,
                                tensorflow_session=session, batch_size=batch_size,
                                data_format='NCHW' if gpu_available else 'NHWC',
                                eye_image_shape=(108, 180))
        else:
            data_source = Webcam(tensorflow_session=session, batch_size=batch_size,
                                 camera_id=args.camera_id, fps=args.fps,
                                 data_format='NCHW' if gpu_available else 'NHWC',
                                 eye_image_shape=(36, 60))



        # Define model

        if args.from_video:
            model = ELG(
                session, train_data={'videostream': data_source},
                first_layer_stride=3,
                num_modules=3,
                num_feature_maps=64,
                learning_schedule=[
                    {
                        'loss_terms_to_optimize': {'dummy': ['hourglass', 'radius']},
                    },
                ],
            )
        else:
            model = ELG(
                session, train_data={'videostream': data_source},
                first_layer_stride=1,
                num_modules=2,
                num_feature_maps=32,
                learning_schedule=[
                    {
                        'loss_terms_to_optimize': {'dummy': ['hourglass', 'radius']},
                    },
                ],
            )

        # Record output frames to file if requested
        if args.record_video:
            video_out = None
            video_out_queue = queue.Queue()
            video_out_should_stop = False
            video_out_done = threading.Condition()

            def _record_frame():
                global video_out
                last_frame_time = None
                out_fps = 30
                out_frame_interval = 1.0 / out_fps
                while not video_out_should_stop:
                    frame_index = video_out_queue.get()
                    if frame_index is None:
                        break
                    assert frame_index in data_source._frames
                    frame = data_source._frames[frame_index]['bgr']
                    h, w, _ = frame.shape
                    if video_out is None:
                        video_out = cv.VideoWriter(
                            args.record_video, cv.VideoWriter_fourcc(*'H264'),
                            out_fps, (w, h),
                        )
                    now_time = time.time()
                    if last_frame_time is not None:
                        time_diff = now_time - last_frame_time
                        while time_diff > 0.0:
                            video_out.write(frame)
                            time_diff -= out_frame_interval
                    last_frame_time = now_time
                video_out.release()
                with video_out_done:
                    video_out_done.notify_all()
            record_thread = threading.Thread(target=_record_frame, name='record')
            record_thread.daemon = True
            record_thread.start()

        # Begin visualization thread
        inferred_stuff_queue = queue.Queue()



        def _visualize_output():

            last_frame_index = 0
            last_frame_time = time.time()
            fps_history = []
            all_gaze_histories = []


            
            if args.fullscreen:
                cv.namedWindow('vis', cv.WND_PROP_FULLSCREEN)
                cv.setWindowProperty('vis', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

            while True:
                global is_detect
                # If no output to visualize, show unannotated frame
                if inferred_stuff_queue.empty():
                    next_frame_index = last_frame_index + 1
                    if next_frame_index in data_source._frames:

                        next_frame = data_source._frames[next_frame_index]
                        # print( 'len : %d  //// faces in : %d ',len(next_frame['faces']), 'faces' in next_frame )
                        if 'faces' in next_frame and len(next_frame['faces']) == 0:


                            is_detect = False

                            if not args.headless:
                                # cv.imshow('vis', next_frame['bgr'])
                                None

                            if args.record_video:
                                video_out_queue.put_nowait(next_frame_index)
                            last_frame_index = next_frame_index

                        elif not 'faces' in next_frame :
                            is_detect = True
                            global start_condition
                            if (start_condition == False):
                                None
                                start_condition = True
                                calibration_thread = threading.Thread(target=start_cali, name='calibration_th2')
                                calibration_thread.daemon = True
                                calibration_thread.start()

                    # print(is_detect) 

                    #/////////////////////////////////////////////////////
                    # if cv.waitKey(1) & 0xFF == ord('q'):
                    #     return
                    #/////////////////////////////////////////////////////
                    continue

                # Get output from neural network and visualize
                output = inferred_stuff_queue.get()
                bgr = None
                for j in range(batch_size):
                    frame_index = output['frame_index'][j]
                    if frame_index not in data_source._frames:
                        continue
                    frame = data_source._frames[frame_index]

                    # Decide which landmarks are usable
                    heatmaps_amax = np.amax(output['heatmaps'][j, :].reshape(-1, 18), axis=0)
                    can_use_eye = np.all(heatmaps_amax > 0.7)
                    can_use_eyelid = np.all(heatmaps_amax[0:8] > 0.75)
                    can_use_iris = np.all(heatmaps_amax[8:16] > 0.8)

                    start_time = time.time()
                    eye_index = output['eye_index'][j]
                    bgr = frame['bgr']
                    eye = frame['eyes'][eye_index]
                    eye_image = eye['image']
                    eye_side = eye['side']
                    eye_landmarks = output['landmarks'][j, :]
                    eye_radius = output['radius'][j][0]
                    if eye_side == 'left':
                        eye_landmarks[:, 0] = eye_image.shape[1] - eye_landmarks[:, 0]
                        eye_image = np.fliplr(eye_image)

                    # Embed eye image and annotate for picture-in-picture
                    eye_upscale = 2
                    eye_image_raw = cv.cvtColor(cv.equalizeHist(eye_image), cv.COLOR_GRAY2BGR)
                    eye_image_raw = cv.resize(eye_image_raw, (0, 0), fx=eye_upscale, fy=eye_upscale)
                    eye_image_annotated = np.copy(eye_image_raw)
                    if can_use_eyelid:
                        cv.polylines(
                            eye_image_annotated,
                            [np.round(eye_upscale*eye_landmarks[0:8]).astype(np.int32)
                                                                     .reshape(-1, 1, 2)],
                            isClosed=True, color=(255, 255, 0), thickness=1, lineType=cv.LINE_AA,
                        )
                    # if can_use_iris:
                    #     cv.polylines(
                    #         eye_image_annotated,
                    #         [np.round(eye_upscale*eye_landmarks[8:16]).astype(np.int32)
                    #                                                   .reshape(-1, 1, 2)],
                    #         isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv.LINE_AA,
                    #     )
                        # cv.drawMarker(
                        #     eye_image_annotated,
                        #     tuple(np.round(eye_upscale*eye_landmarks[16, :]).astype(np.int32)),
                        #     color=(0, 255, 255), markerType=cv.MARKER_CROSS, markerSize=4,
                        #     thickness=1, line_type=cv.LINE_AA,
                        # )
                    face_index = int(eye_index / 2)
                    eh, ew, _ = eye_image_raw.shape
                    v0 = face_index * 2 * eh
                    v1 = v0 + eh
                    v2 = v1 + eh
                    u0 = 0 if eye_side == 'left' else ew
                    u1 = u0 + ew
                    bgr[v0:v1, u0:u1] = eye_image_raw
                    bgr[v1:v2, u0:u1] = eye_image_annotated

                    # Visualize preprocessing results
                    frame_landmarks = (frame['smoothed_landmarks']
                                       if 'smoothed_landmarks' in frame
                                       else frame['landmarks'])


                    for f, face in enumerate(frame['faces']):
                        for landmark in frame_landmarks[f][:-1]:
                            cv.drawMarker(bgr, tuple(np.round(landmark).astype(np.int32)),
                                          color=(0, 0, 255), markerType=cv.MARKER_STAR,
                                          markerSize=2, thickness=1, line_type=cv.LINE_AA)
                        cv.rectangle(
                            bgr, tuple(np.round(face[:2]).astype(np.int32)),
                            tuple(np.round(np.add(face[:2], face[2:])).astype(np.int32)),
                            color=(0, 255, 255), thickness=1, lineType=cv.LINE_AA,
                        )


                    # Transform predictions
                    eye_landmarks = np.concatenate([eye_landmarks, [[eye_landmarks[-1, 0] + eye_radius, eye_landmarks[-1, 1]]]])
                    eye_landmarks = np.asmatrix(np.pad(eye_landmarks, ((0, 0), (0, 1)), 'constant', constant_values=1.0))
                    eye_landmarks = (eye_landmarks * eye['inv_landmarks_transform_mat'].T)[:, :2]
                    eye_landmarks = np.asarray(eye_landmarks)
                    eyelid_landmarks = eye_landmarks[0:8, :]
                    iris_landmarks = eye_landmarks[8:16, :]
                    iris_centre = eye_landmarks[16, :]
                    eyeball_centre = eye_landmarks[17, :]
                    eyeball_radius = np.linalg.norm(eye_landmarks[18, :] -
                                                    eye_landmarks[17, :])

                    # Smooth and visualize gaze direction
                    num_total_eyes_in_frame = len(frame['eyes'])
                    if len(all_gaze_histories) != num_total_eyes_in_frame:
                        all_gaze_histories = [list() for _ in range(num_total_eyes_in_frame)]
                    gaze_history = all_gaze_histories[eye_index]
                    if can_use_eye:
                        # Visualize landmarks
                        cv.drawMarker(  # Eyeball centre
                            bgr, tuple(np.round(eyeball_centre).astype(np.int32)),
                            color=(0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=4,
                            thickness=1, line_type=cv.LINE_AA,
                        )
                        # cv.circle(  # Eyeball outline
                        #     bgr, tuple(np.round(eyeball_centre).astype(np.int32)),
                        #     int(np.round(eyeball_radius)), color=(0, 255, 0),
                        #     thickness=1, lineType=cv.LINE_AA,
                        # )

                        # Draw "gaze"
                        # from models.elg import estimate_gaze_from_landmarks
                        # current_gaze = estimate_gaze_from_landmarks(
                        #     iris_landmarks, iris_centre, eyeball_centre, eyeball_radius)
                        i_x0, i_y0 = iris_centre
                        e_x0, e_y0 = eyeball_centre
                        theta = -np.arcsin(np.clip((i_y0 - e_y0) / eyeball_radius, -1.0, 1.0))
                        phi = np.arcsin(np.clip((i_x0 - e_x0) / (eyeball_radius * -np.cos(theta)),
                                                -1.0, 1.0))
                        current_gaze = np.array([theta, phi])
                        gaze_history.append(current_gaze)
                        gaze_history_max_len = 10
                        
                        if len(gaze_history) > gaze_history_max_len:
                            gaze_history = gaze_history[-gaze_history_max_len:]
                        util.gaze.draw_gaze(bgr, iris_centre, np.mean(gaze_history, axis=0),
                                            length=120.0, thickness=1)
                    else:
                        gaze_history.clear()

                    if can_use_eyelid:
                        cv.polylines(
                            bgr, [np.round(eyelid_landmarks).astype(np.int32).reshape(-1, 1, 2)],
                            isClosed=True, color=(255, 255, 0), thickness=1, lineType=cv.LINE_AA,
                        )

                    if can_use_iris:
                        cv.polylines(
                            bgr, [np.round(iris_landmarks).astype(np.int32).reshape(-1, 1, 2)],
                            isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv.LINE_AA,
                        )
                        cv.drawMarker(
                            bgr, tuple(np.round(iris_centre).astype(np.int32)),
                            color=(0, 255, 255), markerType=cv.MARKER_CROSS, markerSize=4,
                            thickness=1, line_type=cv.LINE_AA,
                        )

                    dtime = 1e3*(time.time() - start_time)
                    if 'visualization' not in frame['time']:
                        frame['time']['visualization'] = dtime
                    else:
                        frame['time']['visualization'] += dtime

                    def _dtime(before_id, after_id):
                        return int(1e3 * (frame['time'][after_id] - frame['time'][before_id]))

                    def _dstr(title, before_id, after_id):
                        return '%s: %dms' % (title, _dtime(before_id, after_id))

                    if eye_index == len(frame['eyes']) - 1:
                        # Calculate timings
                        frame['time']['after_visualization'] = time.time()
                        fps = int(np.round(1.0 / (time.time() - last_frame_time)))
                        fps_history.append(fps)
                        if len(fps_history) > 60:
                            fps_history = fps_history[-60:]
                        fps_str = '%d FPS' % np.mean(fps_history)
                        last_frame_time = time.time()
                        fh, fw, _ = bgr.shape
                        cv.putText(bgr, fps_str, org=(fw - 110, fh - 20),
                                   fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.8,
                                   color=(0, 0, 0), thickness=1, lineType=cv.LINE_AA)
                        cv.putText(bgr, fps_str, org=(fw - 111, fh - 21),
                                   fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.79,
                                   color=(255, 255, 255), thickness=1, lineType=cv.LINE_AA)
                        if not args.headless:


                            # cv.imshow('vis', bgr)
                            None
                        last_frame_index = frame_index

                        # Record frame?
                        if args.record_video:
                            video_out_queue.put_nowait(frame_index)

                        # Quit?
                        # if cv.waitKey(1) & 0xFF == ord('q'):
                        #     return

                        # Print timings
                        if frame_index % 60 == 0:
                            latency = _dtime('before_frame_read', 'after_visualization')
                            processing = _dtime('after_frame_read', 'after_visualization')
                            timing_string = ', '.join([

                                _dstr('read', 'before_frame_read', 'after_frame_read'),
                                _dstr('preproc', 'after_frame_read', 'after_preprocessing'),
                                'infer: %dms' % int(frame['time']['inference']),
                                'vis: %dms' % int(frame['time']['visualization']),
                                'proc: %dms' % processing,
                                'latency: %dms' % latency,
                            ])
                            print('%08d [%s] %s' % (frame_index, fps_str, timing_string))
        
        ## End visualize_output ##

        # calibration_thread = threading.Thread(target=start_cali, name='calibration_th2')
        # calibration_thread.daemon = True
        # calibration_thread.start()


        visualize_thread = threading.Thread(target=_visualize_output, name='visualization')
        visualize_thread.daemon = True
        visualize_thread.start()



        # Do inference forever
        infer = model.inference_generator()


        while True:

            output = next(infer)

            for frame_index in np.unique(output['frame_index']):
                if frame_index not in data_source._frames:
                    continue
                frame = data_source._frames[frame_index]

                if 'inference' in frame['time']:
                    frame['time']['inference'] += output['inference_time']
                else:
                    frame['time']['inference'] = output['inference_time']
            inferred_stuff_queue.put_nowait(output)


            if not visualize_thread.isAlive():
                break

            if not data_source._open:
                break



        # Close video recording
        if args.record_video and video_out is not None:
            video_out_should_stop = True
            video_out_queue.put_nowait(None)
            with video_out_done:
                video_out_done.wait()

