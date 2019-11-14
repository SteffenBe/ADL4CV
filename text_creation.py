import numpy as np

# Placeholders:
# - {shape}: type of shape
# - {col}: fill color of the shape
description_templates = [
    "{col} {shape}",
    "a {col} {shape}",
    "a {shape} that is {col}",
    "a {shape} filled with {col}",
    "make a {col} {shape}",
    "give me a {col} {shape}",
    "give me a {shape} that is {col}",
    "a {col} colored {shape}",
    "the {shape} should be {col}",
]


def get_template_words():
    """Returns a set of all used words in the text templates, excluding placeholders."""
    return set(word for tpl in description_templates
               for word in tpl.split(" ")
               if not word.startswith("{"))


def apply_placeholders(template_str, **kwargs):
    for placeholder, value in kwargs.items():
        template_str = template_str.replace("{"+placeholder+"}", value)
    return template_str


def generate_single_description(shape, fill_color, template_index=None):
    if template_index is None:
        template_index = np.random.choice(len(description_templates))
    return apply_placeholders(description_templates[template_index],
                              shape=shape,
                              col=fill_color)


def generate_descriptions(n, shape, fill_color):
    if n > len(description_templates):
        raise ValueError("cannot generate more than %d unique descriptions" % len(
            description_templates))

    indices = np.random.choice(len(description_templates), n, replace=False)
    return [generate_single_description(shape, fill_color, i) for i in indices]


if __name__ == "__main__":
    n = 5
    shapes = ["square", "triangle"]
    fill_colors = ["blue", "red"]

    np.random.seed(42)
    print("Single:", generate_single_description("square", "blue"))
    print("Multiple:")
    for shape in shapes:
        for fill_color in fill_colors:
            print("----")
            for d in generate_descriptions(n, shape, fill_color):
                print(d)
