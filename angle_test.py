import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = (
        np.arctan2(c[1] - b[1], c[0] - b[0])
        - np.arctan2(a[1] - b[1], a[0] - b[0])
    )

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


# Test points
hip = [100, 100]
knee = [100, 200]
ankle = [100, 300]

angle = calculate_angle(hip, knee, ankle)

print("Knee Angle:", angle)