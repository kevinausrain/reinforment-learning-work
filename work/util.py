import cv2

def car_v2_image_preprocess(img):
    img = img[:84, 6:90]  # CarRacing-v2-specific cropping
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    return img


def atari_v2_image_preprocess(img):
    img = img[8:-12, 4:-12]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    return img