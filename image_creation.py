import numpy as np

from PIL import Image, ImageDraw

import warnings

colors = {"white": (255, 255, 255),
          "black": (0, 0, 0),
          "blue": (0, 0, 255),
          "red": (255, 0, 0)}

image_size = [64, 64]


def save_default_image(image_path="default_image.png", background="white", dimension=image_size):

    image = np.zeros((dimension[0], dimension[0], 3), dtype=np.uint8)

    if background in colors:
        color = colors[background]
        image[:, :, 0] = color[0]
        image[:, :, 1] = color[1]
        image[:, :, 2] = color[2]

    else:
        warning_string = "Warning: Color '%s' not defined in colors dictionary" % background
        warnings.warn(warning_string)

    img = Image.fromarray(image)

    img.save(image_path)


def read_image_to_np(image_path="default_image.png"):

    image = Image.open(image_path)

    np_image = np.array(image)

    return np_image


def calc_traingle_pos(side_length=20, translation=[0, 0], rotation=0, image_dim=image_size):

    center_point = [int(image_dim[0]/2) + translation[0], int(image_dim[0]/2) + translation[1]]

    dist = side_length/np.sqrt(3)

    center_point_math = notation_array_to_point(center_point)

    center_point_math_translated = [center_point_math[0] + translation[0], center_point_math[1] + translation[1]]

    top_point_math = [center_point_math_translated[0], center_point_math_translated[1] + dist]

    point1 = notation_point_to_array(rotate(center_point_math_translated, top_point_math, rotation))
    point2 = notation_point_to_array(rotate(center_point_math_translated, top_point_math, rotation + 120*np.pi/180))
    point3 = notation_point_to_array(rotate(center_point_math_translated, top_point_math, rotation + 240*np.pi/180))

    # point1 = [int(center_point[0] + np.cos(rotation)*side_length), int(center_point[1] + np.sin(rotation)*side_length)]
    # point2 = [int(center_point[0] + np.cos(120 + rotation)*side_length), int(center_point[1] + np.sin(120 + rotation)*side_length)]
    # point3 = [int(center_point[0] + np.cos(240 + rotation)*side_length), int(center_point[1] + np.sin(240 + rotation)*side_length)]

    return [point1, point2, point3]


def calc_square_pos(side_length=20, translation=[0, 0], rotation=0, image_dim=image_size):

    center_point = [int(image_dim[0] / 2) + translation[0], int(image_dim[0] / 2) + translation[1]]

    center_point_math = notation_array_to_point(center_point)

    center_point_math_translated = [center_point_math[0] + translation[0], center_point_math[1] + translation[1]]

    top_left_point = [center_point_math_translated[0] - side_length / 2,
                      center_point_math_translated[1] + side_length / 2]

    bottom_left_point = [center_point_math_translated[0] - side_length / 2,
                         center_point_math_translated[1] - side_length / 2]

    bottom_right_point = [center_point_math_translated[0] + side_length / 2,
                         center_point_math_translated[1] - side_length / 2]

    top_right_point = [center_point_math_translated[0] + side_length / 2,
                      center_point_math_translated[1] + side_length / 2]

    top_l_rotate = notation_point_to_array(rotate(center_point_math_translated, top_left_point, rotation))
    bot_l_rotate = notation_point_to_array(rotate(center_point_math_translated, bottom_left_point, rotation))
    bot_r_rotate = notation_point_to_array(rotate(center_point_math_translated, bottom_right_point, rotation))
    top_r_rotate = notation_point_to_array(rotate(center_point_math_translated, top_right_point, rotation))


    # bottom_left = notation_point_to_array(rotate(center_point_math_translated, top_left_point, rotation + np.pi / 2))
    # bottom_right = notation_point_to_array(rotate(center_point_math_translated, top_left_point, rotation + np.pi))
    # top_right = notation_point_to_array(rotate(center_point_math_translated, top_left_point, rotation + 3 * np.pi / 4))

    # return [notation_point_to_array(top_left_point), bottom_left, bottom_right, top_right]

    return [top_l_rotate, bot_l_rotate, bot_r_rotate, top_r_rotate]


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return [qx, qy]


def notation_array_to_point(array_coord, array_dim=image_size):

    x = array_coord[1]

    y = array_dim[0] - (array_coord[0] + 1)

    return [x, y]


def notation_point_to_array(point_coord, array_dim=image_size):

    x = int(point_coord[0])

    y = int(array_dim[0] - (point_coord[1] + 1))

    return (x, y)

def create_images(folder_path = "images\\", n_per_configuration=1, shapes=["square", "triangle"], fillings=["blue", "red"]):

    base_image = Image.open("default_image.png")

    n_images_total = len(shapes) * len(fillings) * n_per_configuration

    image_number = 0

    for i in range(n_per_configuration):
        for shape in shapes:
            for filling in fillings:

                new_img = base_image.copy()
                translation = np.random.uniform(low=-5, high=5, size=2)
                rotation = np.random.uniform(low=-np.pi / 2, high=np.pi / 2)

                if shape == "square":

                    points = calc_square_pos(translation=translation, rotation=rotation)

                elif shape == "triangle":

                    points = calc_traingle_pos(translation=translation, rotation=rotation)

                draw = ImageDraw.Draw(new_img)

                draw.polygon(points, fill=colors[filling])

                del draw

                img_name = folder_path + "%s_%s_%s.png" %(image_number, shape, filling)

                image_number += 1

                new_img.save(img_name)


if __name__ == "__main__":

    try:
        Image.open("default_image.png")
    except:
        save_default_image()


    create_images(n_per_configuration=5)