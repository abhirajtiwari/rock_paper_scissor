import cv2 as cv
import numpy as np
import tensorflow as tf
import threading
import time

cam = cv.VideoCapture(0)

classes = {
    0: 'Paper',
    1: 'Rock',
    2: 'Scissor'
}

class_no = 0
comp_play = 0

model = tf.keras.models.load_model('models/cus_ds_90_do_0_4.h5')

rock = cv.imread('Images/rock.png')
paper = cv.imread('Images/paper.png')
scissors = cv.imread('Images/scissors.jpg')

rock = cv.resize(rock, (480, 480))
paper = cv.resize(paper, (480, 480))
scissors = cv.resize(scissors, (480, 480))
comp = [paper, rock , scissors]

running = True

def game():
    global running, class_no, comp_play
    while running:
        np.random.seed(int(time.time()))
        comp_play = np.random.randint(3)
        time.sleep(1)
        print('Rock')
        time.sleep(1)
        print('Paper')
        time.sleep(1)
        print('Scissors')

        # cv.imshow('Computer', comp[comp_play])
        # cv.waitKey(0)

        if comp_play-class_no == 1 or comp_play-class_no == -2:
            print('You Won')
        elif comp_play- class_no == 0:
            print('Draw')
        else:
            print('Computer Won!')
        print('\n' + '-'*10 + '\n')
        time.sleep(3)

def prediction_engine():
    global running, class_no, classes
    while True:
        _, frame = cam.read()
        frame = cv.flip(frame, 1)

        cv.rectangle(frame, (100, 100), (400, 400), (255, 255, 255), 2)

        input_image = np.array([frame[100:400, 100:400]])/255.

        class_no = np.argmax(model.predict(input_image), axis=-1)[0]
        predicted_class = classes[class_no]

        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(frame, "{}".format(predicted_class), (5, 50), font, 0.7, (0, 255, 255), 2, cv.LINE_AA)
        
        cv.imshow('Rock Paper Scissor', np.hstack((frame, comp[comp_play])))

        key = cv.waitKey(1)
        if key == ord('q'):
            break
    running = False

try:
    prediction_thread = threading.Thread(target=prediction_engine)
    game_thread = threading.Thread(target=game)
    prediction_thread.start()
    game_thread.start()
except Exception as e:
    print(e)
    print('Cannot start thread')

prediction_thread.join()
game_thread.join()
cam.release()
