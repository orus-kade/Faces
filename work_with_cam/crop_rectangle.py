import dlib


def crop_rectangle(image):
    def find_face_rectangle(image):
        faces_rect = []
        detector = dlib.get_frontal_face_detector()
        face_rects = detector(image, 1)
        left, right, bottom, top = 0, 0, 0, 0
        for i, d in enumerate(face_rects):
            left = d.left()
            top = d.top()
            right = d.right()
            bottom = d.bottom()
            faces.append([left, top, right, bottom])

        if faces[0][0] == 0 and faces[0][1] == 0 and faces[0][2] == 0 and faces[0][3] == 0:
            return None
        if i > 0:
            return None
        else:
            return faces

    def crop(image, faces):
        croped_images = []
        for face in faces:
            croped_image = image.crop((face[0], face[1], face[2], face[3]))
            croped_image = croped_image.resize((128, 128))
            croped_images.append(croped_image)
        return croped_images

    faces = find_face_rectangle(image)
    croped_images = crop(image, faces)
    return croped_images
