import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt 

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test ) = mnist.load_data() 

x_test = tf.keras.utils.normalize(x_test, axis=1)
x_train = tf.keras.utils.normalize(x_train, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics= ['accuracy'])
model.fit(x_train,y_train,epochs = 3)

val_loss, val_acc = model.evaluate(x_test,y_test)
print(val_loss, val_acc)

#model.save('digit_reader.model')

#new_model = tf.keras.models.load_model('digit_reader.model')
predictions = model.predict([x_train])

print(np.argmax(predictions[5]))
plt.imshow(x_test[5])
plt.show()
#!/usr/bin/env python3
import time
from time import sleep
from ev3dev2.button import Button
from ev3dev2.sound import Sound
from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_B, OUTPUT_C, MediumMotor, SpeedPercent, MoveTank
from ev3dev2.sensor import INPUT_1, INPUT_2, INPUT_3, INPUT_4
from ev3dev2.sensor.lego import InfraredSensor, TouchSensor, UltrasonicSensor, ColorSensor, LightSensor, GyroSensor, SoundSensor
from ev3dev2.led import Leds


cs = ColorSensor()
us = UltrasonicSensor()
#ss= SoundSensor()
#ts = TouchSensor()
leds = Leds()
sound = Sound()
btn = Button()
arm = MediumMotor(OUTPUT_C)
tank = MoveSteering(OUTPUT_A, OUTPUT_B)
"""
print("Press the touch sensor to change the LED color!")
while True:
	if ts.is_pressed:
		leds.set_color("LEFT", "RED")
		leds.set_color("RIGHT", "RED")
		sound.speak('Detonation In, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, BOOM!')
	else:
		leds.set_color("LEFT", "GREEN")
		leds.set_color("RIGHT", "GREEN")

print("Press a button to start")

while not btn.any(): # While no (not any) button is pressed.
    sleep(0.001)  # Wait 0.01 second
sleep(5)
"""

def roam():
	tank.on_for_rotations(40, 50, 3, brake=True, block=True)

def attack():
	if us.us.distance_centimeters_ping < 20:
		

def divert():
	pass

def is_detected():
	pass


while not btn.any():
