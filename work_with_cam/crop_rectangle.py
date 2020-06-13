import dlib
from PIL import Image
import numpy as np


def crop_rectangle(image):
    def find_face_rectangle(img):
        detector = dlib.get_frontal_face_detector()
        rectangles = detector(img, 1)
        faces = []
        for i, d in enumerate(rectangles):
            left = d.left()
            top = d.top()
            right = d.right()
            bottom = d.bottom()
            faces.append([left, top, right, bottom])

        if len(faces) == 0:
            return None
        else:
            return faces

    def crop(img, rectangles):
        img_faces = []
        img = Image.fromarray(img)
        for rectangle in rectangles:
            img_for_crop = img.copy()
            img_face = img_for_crop.crop((rectangle[0], rectangle[1], rectangle[2], rectangle[3]))
            img_face = img_face.resize((224, 224))
            img_faces.append(np.array(img_face))
        return img_faces

    small_faces = find_face_rectangle(image)
    if small_faces is None:
        return None
    images = crop(image, small_faces)
    return images
