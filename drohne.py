import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from djitellopy import Tello
from inputs import devices
import cv2
import numpy as np
import time
import utils
from globals import *

S = 100
FPS = 120

class DrohneThread(QThread):
    """Maintains the Tello connection and processes state."""
    # Custom signal to safely pass frames from background thread to Qt GUI
    frame_updated = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.ip = "192.168.0.101"
        self.landed = True
        self.connected = False

        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        self.send_rc_control = False          #erst connecten wenn man auf connect-button drueckt und nicht sobald man die Klasse Drohne instanziert

    def fake_init(self):
        print(f"get ip: {self.ip}")
        self.tello = Tello(host=self.ip)

        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        self.send_rc_control = False
        self.connected = True                 #send stream

    def run(self):
        self.tello.connect(wait_for_state=False)
        self.tello.set_speed(self.speed)
        
        print(self.tello.send_command_with_return("downvision 1"))
        
        self.tello.streamoff()
        self.tello.streamon()

        self.frame_read = self.tello.get_frame_read()

        self.lr, self.fb, self.ud, self.yaw = 0, 0, 0, 0

        while True:                        
            if self.frame_read.stopped:
                print(f"Frame Read Stopped!")
                break
            
            try:
                events = devices.gamepads[0].read()
                for event in events:
                    # Filter for Absolute Axis events (gimbals)
                    if event.ev_type == 'Absolute':
                        self.process_axis(event.code, event.state)

                    elif event.ev_type == 'Key':
                        self.process_button(event.code, event.state)

                if self.tello.is_flying:
                    self.tello.send_rc_control(self.lr, self.fb, self.ud, self.yaw)

            except Exception as e:
                print(f"Controller Error: {e}")
                break


            # Grab the frame and emit it to the main thread
            frame_display = self.frame_read.frame.copy()
            self.frame_updated.emit(frame_display)

            time.sleep(1 / FPS)

        self.tello.end()

    def process_axis(self, axis_code, raw_value):
        """
        Converts hardware axis raw values to Tello's required -100 to 100 range.
        Note: EdgeTX / OpenTX USB HID gimbals typically output values 
        ranging from 0 to 255 (with 127/128 as center) or -32768 to 32767.
        """
        max_gimball_axis = 2048

        normalized = raw_value - (max_gimball_axis // 2)
        scaled = int((normalized / (max_gimball_axis // 2)) * 100)

        print(f"AAAAAAAA:{axis_code}, {scaled}")

        # Deadzone (like on gamepads :) )
        if abs(scaled) < 15:
            scaled = 0

        # Cap boundaries strictly to prevent Tello errors
        scaled = max(-100, min(100, scaled))

        if axis_code == 'ABS_X':
            self.lr = scaled
        if axis_code == 'ABS_Y':
            self.fb = scaled 
        if axis_code == 'ABS_Z':
            self.ud = scaled
        if axis_code == 'ABS_RX':
            self.yaw = scaled

    def process_button(self, button_code, state):
        pass


    def keydown(self, key):
        if key.key() == Qt.Key_Up or key.key() == Qt.Key_U:
            print("Up")
            self.for_back_velocity = S
        elif key.key() == Qt.Key_Down or key.key() == Qt.Key_J:
            print("Down")
            self.for_back_velocity = -S
        elif key.key() == Qt.Key_Left or key.key() == Qt.Key_H:
            print("Left")
            self.left_right_velocity = -S
        elif key.key() == Qt.Key_Right or key.key() == Qt.Key_K:
            print("Right")
            self.left_right_velocity = S
        elif key.key() == Qt.Key_W:
            self.up_down_velocity = S
        elif key.key() == Qt.Key_S:
            self.up_down_velocity = -S
        elif key.key() == Qt.Key_A:
            self.yaw_velocity = -S
        elif key.key() == Qt.Key_D:
            self.yaw_velocity = S
        elif key.key() == Qt.Key_T:
            #if self.allow_takeoff() and self.connected:
            if self.allow_takeoff() and self.connected:
                if abs(self.tello.get_roll()) >= 20 or abs(self.tello.get_pitch()) >= 20:
                    self.send_rc_control = False
                    print("Drohne muss gerade stehen")
                elif self.tello.get_battery() <= 20:
                    self.send_rc_control = False
                    print("Unter 20% Akku")
                else:
                    print("takeoff")
                    self.tello.takeoff()
                    self.send_rc_control = True
        elif key.key() == Qt.Key_L:
            if self.allow_land() and self.connected:
                print("land")
                self.tello.land()
                self.send_rc_control = False
        elif key.key() == Qt.Key_F:
            if self.connected:
                self.keyFoto()

    def keyup(self, key):
        if key.key() == Qt.Key_Up or key.key() == Qt.Key_Down or key.key() == Qt.Key_U or key.key() == Qt.Key_J:
            self.for_back_velocity = 0
        elif key.key() == Qt.Key_Left or key.key() == Qt.Key_Right or key.key() == Qt.Key_H or key.key() == Qt.Key_K:
            self.left_right_velocity = 0
        elif key.key() == Qt.Key_W or key.key() == Qt.Key_S:
            self.up_down_velocity = 0
        elif key.key() == Qt.Key_A or key.key() == Qt.Key_D:
            self.yaw_velocity = 0
            
    def keyFoto(self):
        frame = self.frame_read.frame.copy()
        cv2.imwrite(f"{IMG_DIR}/{IMG}", frame)
        print("Foto gespeichert")
        utils.image_crop()

    def update(self):
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity,
                self.up_down_velocity, self.yaw_velocity)

    def allow_land(self):
        if self.landed:
            return False
        elif not self.landed:
            self.landed = True
            return True

    def allow_takeoff(self):
        if self.landed:
            self.landed = False
            return True
        elif not self.landed:
            return False
