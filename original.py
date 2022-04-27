import matplotlib.pyplot as plt

import json
import csv
import numpy as np
from numpy import array, linspace, radians

# read JSON
f = open("7thStraightupLH.json", )
data = json.load(f)

# naming the x-axis
plt.xlabel('x - axis')
# naming the y-axis
plt.ylabel('y - axis')

# giving a title to my graph
plt.title('Traction Circle!')
# Drawing lines
plt.plot([.6, -.6], [0, 0], [0, 0], [-.6, .6])

# Getting the gravity from the starting point
x_acceleration_initial = data["vectors"][0]["gravity"]["x"]  # lastX
y_acceleration_initial = data["vectors"][0]["gravity"]["y"]  # lastY
z_acceleration_initial = data["vectors"][0]["gravity"]["z"]  # LastZ
recorded_stop = 0
for i in data['vectors']:
    if i["speedMph"] < .5:
        # Getting the gravity from the stopped point
        x_acceleration_initial = i["gravity"]["x"]  # lastX
        y_acceleration_initial = i["gravity"]["y"]  # lastY
        z_acceleration_initial = i["gravity"]["z"]  # LastZ
        recorded_stop = i
        print("Initial x =", i["gravity"]["x"])
        break

# gravitation vector that act upon the phone
gravityVector = array([x_acceleration_initial, y_acceleration_initial, z_acceleration_initial])

# This is the phone orientation relative to downward (positive Z gravitation acceleration) for the cross product
phoneForwardVector = array([-1, 0, 0])

# the cross product of gravitation vector and phone orientation to find out
# the third vector that perpendeicular to other 2 vector (Right hand rule
# to find out her right direction of the phone which will be use for the
# car right direction (car right direction will be the same direction of the phone)
carRightDirection = np.cross(gravityVector, phoneForwardVector)

# another cross product between car right direction and gravitaion vector
# to figure out the direction that the car is moving
carForwardDirection = np.cross(carRightDirection, gravityVector)


# the angle between the 2 vectors function
def angleBetweenVector(v1, v2):
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


max_cornering_x = 0
max_braking = 0
max_accelerating = 0
max_cornering_y = 0
max_magnitude = 0
count = 0
angle_rotation = 0
speed_tracker = 0

with open('7thStraightupLH.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    fields = ['magnitude right', 'magnitudeForward', "speed"]
    writer.writerow(fields)

    for i in data['vectors']:
        speed_tracker = i["speedMph"]
        x = i["acceleration"]["x"]
        y = i["acceleration"]["y"]
        z = i["acceleration"]["z"]
        count += 1

        actual_acceleration = array([x - x_acceleration_initial, y - y_acceleration_initial, z - z_acceleration_initial])

        # angel between the acceleration and car direction vectors
        angle_right = angleBetweenVector(actual_acceleration, carRightDirection)
        #angle_right = angleBetweenVector(carRightDirection, actual_acceleration)

        # The magnitude of acceleration vector
        magnitudeRight = np.cos(angle_right) * np.linalg.norm(actual_acceleration)

        # angle between the acceleration and car forward direction
        angle_straight = angleBetweenVector(actual_acceleration, carForwardDirection)

        # magnitude of the car forward direction
        magnitudeForward = np.cos(angle_straight) * np.linalg.norm(actual_acceleration)

        # if np.sqrt(magnitudeRight**2 + magnitudeForward**2) > max_magnitude:
        #     max_magnitude = np.sqrt(magnitudeRight**2 + magnitudeForward**2)
        #     max_cornering_x = magnitudeRight
        #     max_cornering_y = magnitudeForward
        # loop to find the max cornering
        if max_cornering_x < abs(magnitudeRight):
            max_cornering_x = magnitudeRight
            max_cornering_y = magnitudeForward
            #angle_rotation = np.arctan(max_cornering_y/max_cornering_x)


        # loop to find the max braking
        if max_braking > magnitudeForward:
            max_braking = magnitudeForward

        # loop to find the max acceleration
        if max_accelerating < magnitudeForward:
            max_accelerating = magnitudeForward

        #if i["speedMph"] < 3:
            # plot scatter plot of x and y axis
        plt.scatter(-magnitudeRight,magnitudeForward)
            #plt.scatter((-magnitudeRight*np.cos(-angle_rotation) - (magnitudeForward * np.sin(-angle_rotation))), (magnitudeForward * np.cos(-angle_rotation) + (magnitudeRight * np.sin(-angle_rotation))))

            # write to CSV file
        rows = [[-magnitudeRight, magnitudeForward, speed_tracker]]
        writer.writerows(rows)

    #print(i["acceleration"]["y"])
    #print(i["gravity"]["z"])

# t = linspace(0, 360, 360)
# x = max_cornering_x * np.cos(radians(t))  # major of x-axis
# y = max_braking * np.sin(radians(t))  # major of x-axis
# plt.plot(x, y)


car_class = max_cornering_x + max_accelerating + abs(max_braking)
print("Max Cornering: ", max_cornering_x, " +y:", max_cornering_y)
print("Max Accelerating: ", max_accelerating)
print("Max Braking: ", abs(max_braking))
print("angle of rotation", angle_rotation)

print("Car Class: ", car_class)
plt.show()
