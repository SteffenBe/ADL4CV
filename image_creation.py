import numpy as np

from PIL import Image, ImageDraw

import warnings

colors = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "purple": (255, 0, 255),
    "cyan": (0, 255, 255),
}

default_image_size = [64, 64]


def make_default_image(background="white", dimension=default_image_size):
    image = np.zeros((dimension[0], dimension[0], 3), dtype=np.uint8)

    if background in colors:
        color = colors[background]
        image[:, :, 0] = color[0]
        image[:, :, 1] = color[1]
        image[:, :, 2] = color[2]

    else:
        warning_string = "Warning: Color '%s' not defined in colors dictionary" % background
        warnings.warn(warning_string)

    return Image.fromarray(image)


def save_default_image(image_path="default_image.png", background="white", dimension=default_image_size):
    img = make_default_image(background, dimension)
    img.save(image_path)


def read_image_to_np(image_path="default_image.png"):

    image = Image.open(image_path)

    np_image = np.array(image)

    return np_image


def calc_triangle_pos(side_length=20, translation=[0, 0], rotation=0, image_dim=default_image_size):

    center_point = [int(image_dim[0]/2), int(image_dim[0]/2)]

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


def calc_square_pos(side_length=20, translation=[0, 0], rotation=0, image_dim=default_image_size):

    center_point = [int(image_dim[0] / 2), int(image_dim[0] / 2)]

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

def calc_star_pos(side_length=20, translation=[0, 0], rotation=0, image_dim=default_image_size):

    center_point = [int(image_dim[0] / 2), int(image_dim[0] / 2)]
    center_point_math = notation_array_to_point(center_point)
    center_point_math_translated = [center_point_math[0] + translation[0], center_point_math[1] + translation[1]]
    inner_radius = side_length/(((1 + 5 ** 0.5) / 2) + 1)

    top_point = [center_point_math_translated[0], center_point_math_translated[1] + side_length]
    pentagon_point = rotate(center_point_math_translated,
                            [center_point_math_translated[0], center_point_math_translated[1]+inner_radius],
                            2*np.pi*36/360)

    star_points = [notation_point_to_array(rotate(center_point_math_translated, top_point, rotation+(2*np.pi/5)*i)) for i in range(5)]
    pentagon_points = [notation_point_to_array(rotate(center_point_math_translated, pentagon_point, rotation+(2*np.pi/5)*i)) for i in range(5)]
    points = star_points + pentagon_points

    return [points[i] for i in [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]]

def calc_ellipse_points(side_length=20, translation=[0, 0], image_dim=default_image_size):

    center_point = [int(image_dim[0] / 2), int(image_dim[0] / 2)]

    center_point_math = notation_array_to_point(center_point)

    center_point_math_translated = [center_point_math[0] + translation[0], center_point_math[1] + translation[1]]

    bottom_left_point = [center_point_math_translated[0] - side_length / 2,
                         center_point_math_translated[1] + side_length / 4]

    top_right_point = [center_point_math_translated[0] + side_length / 2,
                       center_point_math_translated[1] - side_length / 4]

    # bottom_left_point = [center_point_math_translated[0] - side_length / 2,
    #                      center_point_math_translated[1] - side_length / 4]
    #
    # top_right_point = [center_point_math_translated[0] + side_length / 2,
    #                    center_point_math_translated[1] + side_length / 4]

    # print(bottom_left_point)
    # print(top_right_point)

    # bottom_left_point = notation_point_to_array(bottom_left_point)
    # top_right_point = notation_point_to_array(top_right_point)

    return [notation_point_to_array(bottom_left_point), notation_point_to_array(top_right_point)]
    # return (bottom_left_point[0], bottom_left_point[1], top_right_point[0], top_right_point[1])

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


def notation_array_to_point(array_coord, array_dim=default_image_size):

    x = array_coord[1]

    y = array_dim[0] - (array_coord[0] + 1)

    return [x, y]


def notation_point_to_array(point_coord, array_dim=default_image_size):

    x = int(point_coord[0])

    y = int(array_dim[0] - (point_coord[1] + 1))

    return (x, y)



def make_image(base_image, shape, filling, image_size=default_image_size, super_sampling=1):
    super_size = (super_sampling * image_size[0], super_sampling * image_size[1])
    new_img = base_image.copy()
    new_img = new_img.resize(super_size)
    translation = np.random.uniform(low=-5, high=5, size=2)
    rotation = np.random.uniform(low=-np.pi / 2, high=np.pi / 2)
    scale = np.random.uniform(low=0.75, high=1.5)

    draw = ImageDraw.Draw(new_img)

    side_length = super_size[0] / 3 * scale
    if shape == "square":
        points = calc_square_pos(side_length, translation=translation, rotation=rotation, image_dim=super_size)
        draw.polygon(points, fill=colors[filling])

    elif shape == "triangle":
        points = calc_triangle_pos(side_length * 1.5, translation=translation, rotation=rotation, image_dim=super_size)
        draw.polygon(points, fill=colors[filling])

    elif shape == "star":
        points = calc_star_pos(side_length, translation=translation, rotation=rotation, image_dim=super_size)
        draw.polygon(points, fill=colors[filling])

    elif shape == "ellipse":
        helper_img = new_img.copy()
        points = calc_ellipse_points(side_length=side_length*1.5, translation=translation, image_dim=super_size)
        draw.ellipse(points, fill=colors[filling])
        new_img = new_img.convert('RGBA')
        new_img = new_img.rotate(360*rotation/(2*np.pi))
        helper_img.paste(new_img, (0,0), new_img)
        new_img = helper_img.copy()
        new_img = new_img.convert('RGB')
        del helper_img

    else:
        raise ValueError("Image creation process not defined for shape '%s'" % shape)


    # draw = ImageDraw.Draw(new_img)
    # draw.polygon(points, fill=colors[filling])
    del draw
    if super_sampling > 1:
        new_img = new_img.resize(image_size, Image.BILINEAR)
    return new_img


# def create_images(folder_path = "images\\", n_per_configuration=1, shapes=["square", "triangle"], fillings=["blue", "red"]):
#
#     base_image = Image.open("default_image.png")
#
#     n_images_total = len(shapes) * len(fillings) * n_per_configuration
#
#     image_number = 0
#
#     for i in range(n_per_configuration):
#         for shape in shapes:
#             for filling in fillings:
#                 new_img = make_image(base_image, shape, filling, super_sampling=4)
#                 img_name = folder_path + "%s_%s_%s.png" %(image_number, shape, filling)
#
#                 image_number += 1
#
#                 new_img.save(img_name)


if __name__ == "__main__":

    a = 2

    # base_image = make_default_image(background="black")
    # base_image = make_default_image()
    #
    # ex_img = make_image(base_image=base_image, shape="ellipse", filling="red", super_sampling=2)
    # ex_img.show()
