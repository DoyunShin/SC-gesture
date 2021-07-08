class dummy: 
    def __init__(self): pass

class storage(Exception):
    def __init__(self):
        self.debug = False
        self.hand = hand(self)
        pass


class hand(Exception):
    def __init__(self, storage):
        import cv2
        self.opencv = cv2
        self.storage = storage

        self.mediapipe_init()
        self.rstset()

        self.img = None
        self.rang = [1100, 500, 750, 100]
        pass

    def rstset(self):
        self.rst = dummy()
        self.rst.x = dummy()
        self.rst.y = dummy()
        self.rst.x.max = -40
        self.rst.x.min = -80
        self.rst.y.max = 0
        self.rst.y.min = -40
        

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
        self.capture.set(self.opencv.CAP_PROP_FRAME_WIDTH, 1366)
        self.capture.set(self.opencv.CAP_PROP_FRAME_HEIGHT, 768)
        self.success, self.image = self.capture.read()
        while True:
            try:
                if not self.capture.isOpened(): break
                self.success, self.image = self.capture.read()
            except AttributeError as e:
                print(e)
                pass
    
    def show(self):
        while True:
            try: 
                tmp = self.success
            except AttributeError as e:
                print(e)
                continue
            if not self.success:
                print("Ignored empty camera frame")
            else:
                image = self.opencv.cvtColor(self.opencv.flip(self.image, 1), self.opencv.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = self.mediapipe.hands_conf.process(image)

                image.flags.writeable = True
                image = self.opencv.cvtColor(image, self.opencv.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mediapipe.drawing.draw_landmarks(image, hand_landmarks, self.mediapipe.hands.HAND_CONNECTIONS)
                
                    rst = dummy()
                    rst.x = dummy()
                    rst.y = dummy()

                    print(len(results.multi_hand_landmarks))
                    for landmarks in results.multi_hand_landmarks:
                        rst.x.max = 0
                        rst.x.min = 1
                        rst.y.max = 0
                        rst.y.min = 1
                        for landmark in landmarks.landmark:
                            if landmark.x > rst.x.max: rst.x.max = landmark.x
                            if landmark.x < rst.x.min: rst.x.min = landmark.x
                            if landmark.y > rst.y.max: rst.y.max = landmark.y
                            if landmark.y < rst.y.min: rst.y.min = landmark.y
                        
                        rst.x.max *= 1366
                        rst.x.min *= 1366
                        rst.y.max *= 768
                        rst.y.min *= 768

                        rst.x.max += self.rst.x.max
                        rst.x.min += self.rst.x.min
                        rst.y.max += self.rst.y.max
                        rst.y.min += self.rst.y.min

                        rst.x.max = int(rst.x.max)
                        rst.x.min = int(rst.x.min)
                        rst.y.max = int(rst.y.max)
                        rst.y.min = int(rst.y.min)
                        if self.storage.debug == True:
                            print(rst.x.max)
                            print(rst.x.min)
                            print(rst.y.max)
                            print(rst.y.min)
                        
                        if rst.x.max < self.rang[0] and rst.y.max < self.rang[1] and rst.x.min > self.rang[2] and rst.y.min > self.rang[3]:
                            self.opencv.rectangle(image, (rst.x.max, rst.y.max), (rst.x.min, rst.y.min), (0,255,0), 3)
                        else:
                            self.opencv.rectangle(image, (rst.x.max, rst.y.max), (rst.x.min, rst.y.min), (0,0,0), 3)
                        
                #print(results.multi_hand_landmarks)
                self.opencv.rectangle(image, (self.rang[0], self.rang[1]), (self.rang[2], self.rang[3]), (255,255,0), 2)
                self.opencv.imshow('MediaPipe Hands', image)
                if self.opencv.waitKey(5) & 0xFF == 27:
                    self.capture.release()
                    break

if __name__ == "__main__":
    print("started")
    main = storage()
    main.hand.capture()
    print("d")
    main.hand.show()
    pass