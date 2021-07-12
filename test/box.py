from typing import Type

from numpy import zeros
import requests


class dummy: 
    def __init__(self): pass

class storage(Exception):
    def __init__(self):
        self.debug = False
        self.test = True

        self.rstset()
        self.load_data()
        self.load_opencv()
        
        self.camera = [600, 480]
        #self.rang = [1100, 500, 750, 100]
        self.rang = [1100/1366, 500/768, 750/1366, 100/768]
        self.tolerance = 0.3
        self.authorized = False
        self.facestart = False
        self.stack = 0
        self.watitime = 2

        
        self.capture = capture(self)
        self.hand = hand(self)
        self.face = face(self)
        self.count = count(self)
        pass

    def rstset(self):
        self.rst = dummy()
        self.rst.x = dummy()
        self.rst.y = dummy()
        self.rst.x.max = -20
        self.rst.x.min = -80
        self.rst.y.max = 0
        self.rst.y.min = -40

        #RPI
        self.rst.x.max = 50
        self.rst.x.min = 0
        self.rst.y.max = 20
        self.rst.y.min = -20





    def load_opencv(self):
        import cv2
        self.opencv = cv2
    
    def load_data(self):
        from os import path

        if path.isfile("data.pickle"):
            f = open("data.pickle", "rb")
            from pickle import loads
            try:
                print("Load start")
                self.data = loads(f.read())
                print("Load end")
            except TypeError as e:
                print(e)
                self.data = None
        else:
            self.data = None

        return False
        pass
   
    def save_data(self):
        f = open("data.pickle", "wb")
        from pickle import dumps
        f.write(dumps(self.data))
        f.close()

class capture(Exception):
    def __init__(self, storage):
        self.storage = storage
        self.train = dummy()
        self.train.status = False
        self.train.time = None

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
        self.capture = self.storage.opencv.VideoCapture(-1)
        #self.capture.set(self.storage.opencv.CAP_PROP_FRAME_WIDTH, self.storage.camera[0])
        #self.capture.set(self.storage.opencv.CAP_PROP_FRAME_HEIGHT, self.storage.camera[1])
        self.success, self.image = self.capture.read()
        while True:
            try:    
                if not self.capture.isOpened(): break
                self.success, image = self.capture.read()
                self.image = self.storage.opencv.flip(image, 1)
            except AttributeError as e:
                print(e)
                pass

    def show(self):
        while True:
            while True:
                try:
                    tmp = self.success
                    break
                except AttributeError as e:
                    if self.storage.debug: print("DEBUG: "+str(e))
                    pass

            if not self.success and self.storage.debug: print("DEBUG: Ignored empty camera frame")
            else:
                image = self.task(self.image)
                self.storage.opencv.imshow('MPH', image)
                if self.storage.opencv.waitKey(5) & 0xFF == 27:
                    self.capture.release()
                    break

    def task(self, image):
        now = "Unknown"
        if self.train.status:
            print("TRAIN")
            from threading import Thread
            from time import time
            if self.storage.face.train(image) == False: self.train.time+1
            #Thread(target=self.storage.face.train, args=(image,)).start()
            if int(time()) > (self.train.time+2):
                self.train.status = False
                self.train.time = None
                self.storage.save_data()
            now = "FaceTrain"

        elif self.storage.data == None:
            print("TRAIN-INIT")
            from threading import Thread
            from time import time
            self.train.status = True
            self.train.time = int(time())
            if self.storage.face.train(image) == False: self.train.time+1
            #Thread(target=self.storage.face.train, args=(image,)).start()
            now = "FaceTrain"
        else:
            #self.
            if self.storage.facestart == False:
                from threading import Thread
                Thread(target=self.task_recog).start()
                self.storage.facestart = True

            if self.storage.authorized: 
                image, count = self.storage.hand.handcheck(image)
                if count != -1:
                    self.storage.count.main(count)
            else: print("No face.")

        
        self.storage.opencv.putText(image, now, (1,1), self.storage.opencv.FONT_ITALIC, 3, self.storage.opencv.LINE_AA)

        return image
        pass

    def task_recog(self):
        from time import sleep, time
        now = 0
        while True:
            now = int(time())
            if not self.capture.isOpened(): break
            image = self.image
            check, image = self.storage.face.recog(image)
            self.storage.authorized = check
            sleep(5)
        pass


class count(Exception):
    def __init__(self, storage):
        from requests import get
        self.storage = storage
        self.now = "menu"
        self.before = -1
        self.beforetime = 0
        self.req = get
        self.fan = 0
        self.light = 0
        self.fanwork = False
        self.lightwork = False
        pass
    
    def main(self, count):
        from time import time
        

        if count == -1: 
            self.before = -1
            return
        if self.before != count:
                self.beforetime = int(time())
        elif self.before == count and self.beforetime+self.storage.waittime > time():
            if count == 0: self.zero()
            elif count == 1: self.one()
            elif count == 2: self.two()
            elif count == 3: self.three()
            elif count == 4: self.four()
            elif count == 0: self.five()

            self.before = -1
                
        
    def fanc(self, count):
        if self.fanwork == True: return
        self.fanwork = True
        from time import sleep
        if count == 0:
            if self.fan == 0: pass
            elif self.fan == 1:
                self.req("http://192.168.4.1/H")
                sleep(0.7)
                self.req("http://192.168.4.1/H")
                sleep(0.7)
                self.req("http://192.168.4.1/H")
            elif self.fan == 2:
                self.req("http://192.168.4.1/H")
                sleep(0.7)
                self.req("http://192.168.4.1/H")
            elif self.fan == 3:
                self.req("http://192.168.4.1/H")
        elif count == 1:
            if self.fan == 0:
                self.req("http://192.168.4.1/H")
            elif self.fan == 1: pass
            elif self.fan == 2:
                self.req("http://192.168.4.1/H")
                sleep(0.7)
                self.req("http://192.168.4.1/H")
                sleep(0.7)
                self.req("http://192.168.4.1/H")
            elif self.fan == 3:
                self.req("http://192.168.4.1/H")
                sleep(0.7)
                self.req("http://192.168.4.1/H")
        elif count == 2:
            if self.fan == 0:
                self.req("http://192.168.4.1/H")
                sleep(0.7)
                self.req("http://192.168.4.1/H")
            elif self.fan == 1:
                self.req("http://192.168.4.1/H")
            elif self.fan == 2: pass
            elif self.fan == 3:
                self.req("http://192.168.4.1/H")
                sleep(0.7)
                self.req("http://192.168.4.1/H")
                sleep(0.7)
                self.req("http://192.168.4.1/H")
        elif count == 3:
            if self.fan == 0:
                self.req("http://192.168.4.1/H")
                sleep(0.7)
                self.req("http://192.168.4.1/H")
                sleep(0.7)
                self.req("http://192.168.4.1/H")
            elif self.fan == 1:
                self.req("http://192.168.4.1/H")
                sleep(0.7)
                self.req("http://192.168.4.1/H")
            elif self.fan == 2:
                self.req("http://192.168.4.1/H")
            elif self.fan == 3: pass
        else:
            print("SOMETHING WENT WRONG")
        
        self.fan = count
        self.fanwork = False
        
        return

    def lightc(self, count):
        if self.lightwork == True: return
        self.lightwork = True
        if count == 0:
            if self.light == 0: pass
            elif self.light == 1:
                self.req("http://192.168.4.1/L")
        elif count == 1:
            if self.light == 0:
                self.req("http://192.168.4.1/H")
            elif self.light == 1: pass
        else:
            print("SOMETHING WENT WRONG")
        
        self.light = count
        self.lightwork = False
        return


    def zero(self):
        from threading import Thread
        if self.now == "menu": return
        elif self.now == "fan":
            Thread(target=self.fanc, args=(0,)).start()
        elif self.now == "light":
            Thread(target=self.lightc, args=(1,)).start()
            pass
        pass

    def one(self):
        from threading import Thread
        if self.now == "menu":
            self.now = "fan"
        elif self.now == "fan":
            Thread(target=self.fanc, args=(1,)).start()
        elif self.now == "light":
            Thread(target=self.lightc, args=(1,)).start()
            pass
        
        pass

    def two(self):
        from threading import Thread
        if self.now == "menu":
            self.now = "light"
        elif self.now == "fan":
            if self.fanwork == True: return
            Thread(target=self.fanc, args=(2,)).start()
        
        pass

    def three(self):
        from threading import Thread
        if self.now == "menu":
            return
        elif self.now == "fan":
            if self.fanwork == True: return
            Thread(target=self.fanc, args=(3,)).start()
        pass

    def four(self):
        if self.now == "menu":
            return
        pass
    
    def five(self):
        if self.now == "menu": return
        else:
            self.now = "menu"
        pass



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
        self.camera = self.storage.camera
        pass

    def mediapipe_init(self):
        import mediapipe
        self.mediapipe = dummy()
        self.mediapipe.drawing = mediapipe.solutions.drawing_utils
        self.mediapipe.hands = mediapipe.solutions.hands
        self.mediapipe.hands_conf = self.mediapipe.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def handcheck(self, image):
        image = self.storage.opencv.cvtColor(image, self.storage.opencv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.mediapipe.hands_conf.process(image)
        image.flags.writeable = True
        image = self.storage.opencv.cvtColor(image, self.storage.opencv.COLOR_RGB2BGR)
        rclist = []
        count = -1
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
                    rclistb.append({"x": int(landmark.x * self.storage.camera[0]), "y": int(landmark.y * self.storage.camera[1]), "z": landmark.z})
                    if landmark.x > rst.x.max: rst.x.max = landmark.x
                    if landmark.x < rst.x.min: rst.x.min = landmark.x
                    if landmark.y > rst.y.max: rst.y.max = landmark.y
                    if landmark.y < rst.y.min: rst.y.min = landmark.y
                
                rst.x.max *= self.storage.camera[0]
                rst.x.min *= self.storage.camera[0]
                rst.y.max *= self.storage.camera[1]
                rst.y.min *= self.storage.camera[1]

                rst.x.max += self.rst.x.max
                rst.x.min += self.rst.x.min
                rst.y.max += self.rst.y.max
                rst.y.min += self.rst.y.min

                rst.x.max = int(rst.x.max)
                rst.x.min = int(rst.x.min)
                rst.y.max = int(rst.y.max)
                rst.y.min = int(rst.y.min)
                if self.storage.debug:
                    print(rst.x.max)
                    print(rst.x.min)
                    print(rst.y.max)
                    print(rst.y.min)
                
                if rst.x.max < self.rang[0]*self.camera[0] and rst.y.max < self.rang[1]*self.camera[1] and rst.x.min > self.rang[2]*self.camera[0] and rst.y.min > self.rang[3]*self.camera[1]:
                    self.storage.opencv.rectangle(image, (rst.x.max, rst.y.max), (rst.x.min, rst.y.min), (0,255,0), 3)
                    rclist.append({"inbox": True, "data": rclistb})
                    count += 1
                else:
                    self.storage.opencv.rectangle(image, (rst.x.max, rst.y.max), (rst.x.min, rst.y.min), (0,0,0), 3)
                    rclist.append({"inbox": False, "data": rclistb})
                    pass

            if count == 1:
                if self.storage.test:
                    image, count = self.hand_int(image, rclist, rst)
                    pass

                pass
            elif count == 2:
                pass

            if self.storage.debug: print(results.multi_hand_landmarks)

        self.storage.opencv.rectangle(image, (int(self.rang[0]*self.camera[0]), int(self.rang[1]*self.camera[1])), (int(self.rang[2]*self.camera[0]), int(self.rang[3]*self.camera[1])), (255,255,0), 2)

        return image, count

    def hand_int(self, image, result, rst):
        if result[0]["inbox"]: data = result[0]["data"]
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
        if self.storage.debug: print(count)

        return image, count
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

        rgb = self.storage.opencv.cvtColor(image, self.storage.opencv.COLOR_BGR2RGB)
        boxes = self.face_recognition.face_locations(rgb)
        encodings = self.face_recognition.face_encodings(rgb, boxes)

        if encodings == []: return False
        for encoding in encodings:
            self.storage.data["encodings"].append(encoding)
            self.storage.data["names"].append("Authorized")

        return True
        pass

    def recog(self, image): 
        import numpy as np
        self.checkdata()

        image_rgb = self.storage.opencv.cvtColor(image, self.storage.opencv.COLOR_BGR2RGB)
        location = self.face_recognition.face_locations(image_rgb)
        encodings = self.face_recognition.face_encodings(image_rgb, location)

        names = []
        dtc = False

        for encoding in encodings:
            matches = self.face_recognition.compare_faces(self.storage.data["encodings"], encoding)
            name = "Unknown"

            face_distances = self.face_recognition.face_distance(self.storage.data["encodings"], encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]: 
                name = self.storage.data["names"][best_match_index]
                dtc = True

            names.append(name)

        return dtc, image
        pass

if __name__ == "__main__":
    print("started")
    main = storage()
    main.capture.start()
    main.capture.show()
    pass
