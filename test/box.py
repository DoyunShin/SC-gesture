class dummy: 
    def __init__(self): pass

class storage(Exception):
    def __init__(self):
        self.debug = False
        self.test = True

        self.rstset()
        self.load_data()
        self.load_opencv()

        self.rang = [1100, 500, 750, 100]
        self.tolerance = 0.3

        self.capture = capture(self)
        self.hand = hand(self)
        pass

    def rstset(self):
        self.rst = dummy()
        self.rst.x = dummy()
        self.rst.y = dummy()
        self.rst.x.max = -20
        self.rst.x.min = -80
        self.rst.y.max = 0
        self.rst.y.min = -40





    def load_opencv(self):
        import cv2
        self.opencv = cv2
    
    def load_data(self):
        from os import path

        if path.isfile("data.json") and path.exists("data.json"):
            f = open("data.json", "r")
            from json import loads
            self.data = loads(f)
        else:
            self.storage = None

        return False
        pass
   
    def save_data(self):
        f = open("data.json", "w")
        from json import dump
        f.write(dump(self.data))
        f.close()

class capture(Exception):
    def __init__(self, storage):
        self.storage = storage

        try:
            tmp = self.storage.opencv
        except AttributeError:
            self.storage.load_opencv()
        pass

    def start(self):
        from threading import Thread
        self.capturethread = Thread(target=self.capture_thread)
        self.capturethread.start()


    def capture_thread(self):
        self.capture = self.storage.opencv.VideoCapture(0)
        self.capture.set(self.storage.opencv.CAP_PROP_FRAME_WIDTH, 1366)
        self.capture.set(self.storage.opencv.CAP_PROP_FRAME_HEIGHT, 768)
        self.success, self.image = self.capture.read()
        while True:
            try:    
                if not self.storage.capture.capture.isOpened(): break
                self.success, self.image = self.capture.read()
            except AttributeError as e:
                print(e)
                pass

    def show(self):
        while True:
            while True:
                try:
                    tmp = self.capture.success
                    break
                except AttributeError:
                    if self.storage.debug == True: print("DEBUG: "+str(e))
                    pass

            if not self.success and self.storage.debug == True: print("DEBUG: Ignored empty camera frame")
            else:
                image = self.storage.opencv.cvtColor(self.storage.opencv.flip(self.storage.capture.image, 1), self.storage.opencv.COLOR_BGR2RGB)
                image = self.storage.hand.handcheck(image)
                self.storage.opencv.imshow('MPH', image)
                if self.storage.opencv.waitKey(5) & 0xFF == 27:
                    self.capture.release()
                    break


class hand(Exception):
    def __init__(self, storage):
        self.storage = storage

        try:
            tmp = self.storage.opencv
        except AttributeError:
            self.storage.load_opencv()


        self.mediapipe_init()
        self.rst = self.storage.rst

        self.rang = self.storage.rang
        pass

    def mediapipe_init(self):
        import mediapipe
        self.mediapipe = dummy()
        self.mediapipe.drawing = mediapipe.solutions.drawing_utils
        self.mediapipe.hands = mediapipe.solutions.hands
        self.mediapipe.hands_conf = self.mediapipe.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def handcheck(self, image):
        image.flags.writeable = False
        results = self.mediapipe.hands_conf.process(image)

        image.flags.writeable = True
        image = self.storage.opencv.cvtColor(image, self.storage.opencv.COLOR_RGB2BGR)
        rclist = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mediapipe.drawing.draw_landmarks(image, hand_landmarks, self.mediapipe.hands.HAND_CONNECTIONS)

            count = 0
            rst = dummy()
            rst.x = dummy()
            rst.y = dummy()

            for landmarks in results.multi_hand_landmarks:
                rclistb = []
                inbox = False
                rst.x.max = 0
                rst.x.min = 1
                rst.y.max = 0
                rst.y.min = 1
                for landmark in landmarks.landmark:
                    rclistb.append({"x": int(landmark.x * 1366), "y": int(landmark.y * 768), "z": landmark.z})
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
                    self.storage.opencv.rectangle(image, (rst.x.max, rst.y.max), (rst.x.min, rst.y.min), (0,255,0), 3)
                    rclist.append({"inbox": True, "data": rclistb})
                    count += 1
                else:
                    self.storage.opencv.rectangle(image, (rst.x.max, rst.y.max), (rst.x.min, rst.y.min), (0,0,0), 3)
                    rclist.append({"inbox": False, "data": rclistb})
                    pass

            if count == 1:
                if self.storage.test == True:
                    image = self.hand_int(image, rclist, rst)
                    pass

                pass
            elif count == 2:
                pass

            if self.storage.debug == True: print(results.multi_hand_landmarks)

        self.storage.opencv.rectangle(image, (self.rang[0], self.rang[1]), (self.rang[2], self.rang[3]), (255,255,0), 2)

        return image

    def hand_int(self, image, result, rst):
        if result[0]["inbox"] == True: data = result[0]["data"]
        else: data = result[1]["data"]

        count = 0
        mid = data[9]["y"]-20
        if data[5]["x"] > data[9]["x"]: # 왼손
            if data[4]["x"] < data[3]["x"]: pass
            elif data[4]["x"] > data[5]["x"]: count += 1
        else: # 오른손
            if data[4]["x"] > data[3]["x"]: pass
            elif data[4]["x"] < data[5]["x"]: count += 1
        
        if mid > data[20]["y"]: count += 1
        if mid > data[16]["y"]: count += 1
        if mid > data[12]["y"]: count += 1
        if mid > data[8]["y"]: count += 1

        self.storage.opencv.putText(image, str(count), (rst.x.min, rst.y.max), self.storage.opencv.FONT_ITALIC, 2, self.storage.opencv.LINE_AA)

        #print(str(count)+", "+str(data[8]["y"])+", "+str(mid))
        print(count)

        return image
        pass

class face(Exception): 
    def __init__(self, storage):
        import face_recognition
        self.face_recognition = face_recognition
        self.storage = storage
        self.storage.train = self

        try:
            tmp = self.storage.opencv
        except AttributeError:
            self.storage.load_opencv()

        pass

    def checkdata(self):
        try:
            if self.storage.data == None:
                if self.storage.load_data() == False:
                    self.storage.data = {"encodings": [], "names": []}    
        except AttributeError as e:
            if self.storage.load_data() == False:
                self.storage.data = {"encodings": [], "names": []}


    def train(self, image):

        self.checkdata()
        self.storage.data = {"encodings": [], "names": []}

        rgb = self.storage.opencv.cvtColor(image, self.storage.opencv.COLOR_BGR2RGB)
        boxes = self.face_recognition.face_locations(rgb, model="hog")
        encodings = self.face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            self.storage.data["encodings"].append(encoding)
            self.storage.data["names"].append("Authorized")

        
        pass

    def recog(self, image): 
        self.checkdata()

        rgb = self.storage.opencv.cvtColor(image, self.storage.opencv.COLOR_BGR2RGB)
        boxes = self.face_recognition.face_locations(rgb, model="hog")
        encodings = self.face_recognition.face_encodings(rgb, boxes)

        names = []

        for encoding in encodings:
            matches = self.face_recognition.compare_faces(self.storage.data["encodings"], encoding, tolerance=self.storage.tolerance)

            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                for i in matchedIdxs:
                    name = self.storage.data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)

        for ((top, right, bottom, left), name) in zip(boxes, names):
            #rect_HSize = right - left
            #rect_VSize = right - left
            # draw the predicted face name on the image - color is in BGR
            if name == "Authorized":
                self.storage.opencv.rectangle(image, (left, top), (right, bottom),(0, 255, 0), 2)
            else:
                self.storage.opencv.rectangle(image, (left, top), (right, bottom),(255, 0, 0), 2)


        pass

if __name__ == "__main__":
    print("started")
    main = storage()
    main.capture.start()
    main.capture.show()
    pass
