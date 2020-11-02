import cv2
import os
import sys
import random
import math
import re
import time
import warnings
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import subprocess
import numpy as np
import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.coco import widerface


def grab_frame(cap):
    ret,frame = cap.read()
    if ret!=0:
        return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

def do_detect(frame, model, dataset):
    t1 = time.time()
    results = model.detect([frame], verbose=1)
    print('Detection time is {}'.format(time.time() - t1))
    r = results[0]
    print(r)
    boxes = r['rois']
    N = boxes.shape[0]
    for i in range(N):
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i].tolist()  #convert numpy.ndarray to list
        #TODO: randomize colors
        #random_colors_tuple = tuple(int(el) for el in visualize.random_colors(N)[i])[::-1] #int tuple BGR
        colors_tuple = (0,0,0)

        #print('random_colors_tuple:',random_colors_tuple)
        class_id = r['class_ids'][i]
        score = r['scores'][i]
        label = dataset.class_names[class_id]
        text = "{} {:.3f}".format(label, score) if score else label
        frame = cv2.rectangle(img=frame, pt1=(x1, y1), pt2=(x2, y2), color=colors_tuple, thickness=2)
        frame = cv2.putText(img=frame, text=text, org=(x1, y2), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=(0, 0, 255), thickness=1)
    return frame


def main():

    parser = argparse.ArgumentParser('face detection in realtime.')
    parser.add_argument('--weights_path', default="../logs/widerface20201008T2142/mask_rcnn_widerface_0160.h5",
                    help='path to trained weights.')
    parser.add_argument('--video_pointer', default=0, help='(live) camera number or path to video file')
    parser.add_argument('--video_resolution', default=[1280, 720],
                    help='camera resolution, check available resolutions by command: uvcdynctrl -f')
    parser.add_argument('--restore_audio', default=True, type=bool, help='True if preservation of original audio is desired')
    args = parser.parse_args()



    #convert video_pointer to int if possible
    try:
        video_pointer = int(args.video_pointer)
    except:
        video_pointer = args.video_pointer
        video_name = video_pointer.split('/')[-1]
        video_nameNoExt = video_name.split('.')[0]

    weights_path=args.weights_path
    config = widerface.CocoConfig()
    WIDERFACE_DIR = "../datasets/widerface"

    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()
    dataset = widerface.CocoDataset()
    dataset.class_names = ['BG', 'face']
    #Currently no need for loading an entire dataset
    #dataset.load_coco(WIDERFACE_DIR, "val")
    # Must call before using the dataset
    #dataset.prepare()

    # Device to load the neural network on.
    # Useful if you're training a model on the same
    # machine, in which case use CPU and leave the
    # GPU for training.
    DEVICE = "/gpu:0"

    #cfg = tf.ConfigProto()
    #session = tf.Session(config=cfg)

    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                  config=config)

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)



    cap = cv2.VideoCapture(video_pointer)
    if isinstance(video_pointer, int):
        print('Camera resolution set at: {}x{}. Capturing frames...'.format(args.video_resolution[0],args.video_resolution[1]))
        #set webcam resolution, check available under: uvcdynctrl -f. If unavailable may fail SILENTLY!
        cap.set(3,args.video_resolution[0])
        cap.set(4,args.video_resolution[1])
        if not cap.isOpened():
            warnings.warn('No camera feed, check --video_pointer, --video_resolution and USB connection')
            cap.release()
            cv2.destroyAllWindows()
        else:
            while cap.isOpened():
                t2 = time.time()
                ret, frame = cap.read()
                #print('ret:',ret)
                if ret:
                    frame = do_detect(frame, model, dataset)

                    #Display frame
                    cv2.imshow('video', frame)  #(window-name, img)
                    print('Time between frames is {}'.format(time.time() - t2))

                    #waitkey(1) waits 1 ms, then returns 32bit int corresponding to the pressed key or -1
                    #ord('q') reutrn unicode value of 'q'
                    #0xFF (10:255) sets left 24 bit to 0 and right 8bits to 1
                    if cv2.waitKey(1) & 0xFF == ord('q'):  #press q to exit
                        break
                else:
                    warnings.warn('No camera feed, check --video_pointer and USB connection')
                    break

            # When everything is done, release the capture
            cap.release()
            cv2.destroyAllWindows()
    elif isinstance(video_pointer, str):
        suddenExit = True #bool to check if user pressed q and closed program while it was still running
        w = int(cap.get(3))
        h = int(cap.get(4))
        print ('w, h, fps: ', int(cap.get(3)),int(cap.get(4)),cap.get(5))
        fps_rate = cap.get(5)
        cap = cv2.VideoCapture(video_pointer)
        #filename='../test_videos/output_'+video_pointer+'.mp4'
        #fourcc=cv2.VideoWriter_fourcc(*'XVID') 'MP4V'
        filename = '../test_videos/output_'+video_nameNoExt+'.mp4'
        print('filename: ',filename)
        out = cv2.VideoWriter(filename=filename, fourcc=cv2.VideoWriter_fourcc(*'MP4V'), fps=fps_rate, frameSize=(w, h))
        while cap.isOpened():
            t2 = time.time()
            ret, frame = cap.read()
            #print('ret:',ret)
            if ret:
                frame = do_detect(frame, model, dataset)
                out.write(frame)
                cv2.imshow('video_frames', frame)  #(window-name, img)
                print('Time between frames is {}'.format(time.time() - t2))
                if cv2.waitKey(1) & 0xFF == ord('q'):  #press q to exit
                    warnings.warn('Video {} with detection incomplete! Program closed prematurely.'.format(filename))
                    break
            else:
                print('End of video file. Check if detection has been successful')
                suddenExit = False
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        if args.restore_audio and not suddenExit:
            # ffmpeg -i output_short_pull.mp4 -i short_pull.mp4 -c copy -map 0:v:0 -map 1:a:0 -shortest short_pull_final.mp4
            finalFilePath = os.path.join('../test_videos', video_nameNoExt + '_final' + '.mp4')
            process1 = subprocess.run(args=['ffmpeg','-i', filename, '-i', video_pointer, '-c', 'copy', '-map', '0:v:0', '-map',
                                            '1:a:0', '-shortest', finalFilePath], stdout=subprocess.PIPE)
            if process1.returncode==0:
                subprocess.run(['rm', '-rf', filename])
                print('File with detection, but without audio {} successfully removed'.format(filename))
                print('Video file with voice and detection created: {}'.format(finalFilePath))
            else:
                warnings.warn('Something\'s gone wrong probably during combining video from {} and audio from {}. \
                    Investigate the issue, especially check if ffmpeg package is installed.'.format(fileName,video_pointer))


if __name__ == '__main__':
    main()
