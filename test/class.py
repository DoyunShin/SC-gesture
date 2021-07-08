class dummy: 
    def __init__(self): pass

class storage(Exception):
    def __init__(self):

        self.hand = hand(self)
        pass


class hand(Exception):
    def __init__(self, storage):
        import cv2
        self.exit = False
        self.opencv = cv2
        self.storage = storage

        self.mediapipe_init()

        self.img = None
        pass

    def capture(self):
        from threading import Thread
        self.capturethread = Thread(target=self.capture_thread)
        self.capturethread.start()

    def mediapipe_init(self):
        import mediapipe
        self.mediapipe = dummy()
        self.mediapipe.drawing = mediapipe.solutions.drawing_utils
        self.mediapipe.hands = mediapipe.solutions.hands
        self.mediapipe.hands_conf = self.mediapipe.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def capture_thread(self):
        self.capture = self.opencv.VideoCapture(0)
        self.success, self.image = self.capture.read()
        while True:
            try:
                if not self.capture.isOpened(): break
                self.success, self.image = self.capture.read()
            except AttributeError as e:
                print(e)
                pass
    
    def show(self):
        from os import system
        while True:
            try: 
                tmp = self.success
            except AttributeError as e:
                print(e)
                continue
            print(self.success)
            if not self.success:
                print("Ignored empty camera frame")
                print("bc")
            else:
                print("bz")
                image = self.opencv.cvtColor(self.opencv.flip(self.image, 1), self.opencv.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = self.mediapipe.hands_conf.process(image)

                image.flags.writeable = True
                image = self.opencv.cvtColor(image, self.opencv.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mediapipe.drawing.draw_landmarks(image, hand_landmarks, self.mediapipe.hands.HAND_CONNECTIONS)
                self.opencv.imshow('MediaPipe Hands', image)
                if self.opencv.waitKey(5) & 0xFF == 27:
                    self.capture.release()
                    self.exit = True
                    break
        


if __name__ == "__main__":
    print("started")
    main = storage()
    main.hand.capture()
    print("d")
    main.hand.show()
    pass