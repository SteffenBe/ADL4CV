import itertools
import numpy as np
from typing import List, Set

# Placeholders:
# - {instruction}: a random string from the instructions array below
# - {shape}: type of shape
# - {col}: fill color of the shape
description_templates = [
    "{col} {shape}",
    "{col} and a {shape}",
    "{instruction} a {col} {shape}",
    "{instruction} a {col} colored {shape}",
    "{shape} is a shape best drawn with {col}",
    "a {col} {shape}",
    "a {col} colored {shape}",
    "a {shape} filled with {col}",
    "a {shape} that is {col}",
    "a {shape} with {col} filling",
    "a {shape} with {col} ink",
    "fill the {shape} with {col}",
    "fill the {shape} with the color {col}",
    "i need this {col} {shape} in my life",
    "the {shape} should be {col}",
    "the color {col} is perfect for this {shape}",
    "this {shape} should be {col}",
    "use {col} ink for this {shape}",
]

instructions = [
    "draw",
    "first i need",
    "generate",
    "give me",
    "i dream of",
    "i like",
    "i need",
    "i want",
    "i would really enjoy",
    "just draw",
    "just give me",
    "make",
    "paint",
    "please draw",
    "please generate",
    "please give me",
    "please make",
    "show me",
    "show",
]

modifications_shape = [
    "make it a {shape}",
    "change it to a {shape}",
    "i prefer a {shape}",
    "i would prefer a {shape}",
    "a {shape} is better",
    "a {shape} would be better",
]

modifications_color = [
    "make it {col}",
    "please in {col}",
    "change the color to {col}",
    "i prefer {col}",
    "i would prefer {col}",
    "try it in {col}",
    "{col} would be better",
    "{col} please",
]


def get_template_words() -> Set[str]:
    """Returns a set of all used words in the text templates, excluding placeholders."""
    all_templates = instructions + description_templates + modifications_shape + modifications_color
    return set(word for tpl in all_templates
               for word in tpl.split(" ")
               if not word.startswith("{"))


def apply_placeholders(template_str: str, **kwargs) -> str:
    for placeholder, value in kwargs.items():
        template_str = template_str.replace("{"+placeholder+"}", value)
    return template_str


def generate_single_modification(*, new_shape: str = None, new_color: str = None) -> str:
    if not new_shape and not new_color:
        raise ValueError("need to specify either new_shape or new_color")
    if new_shape and new_color:
        raise ValueError("cannot specify both new_shape and new_color")

    if new_shape:
        template = np.random.choice(modifications_shape)
        return apply_placeholders(template, shape=new_shape)
    else:
        template = np.random.choice(modifications_color)
        return apply_placeholders(template, col=new_color)


def generate_single_description(shape: str, fill_color: str, template: str = None) -> str:
    if template is None:
        template = np.random.choice(description_templates)
    return apply_placeholders(template,
                              instruction=np.random.choice(instructions),
                              shape=shape,
                              col=fill_color)


_baked_description_templates = None
_baked_description_weights = None


def bake_descriptions():
    global _baked_description_templates, _baked_description_weights
    if _baked_description_templates is not None and _baked_description_weights is not None:
        return
    baked_nums = {}
    for tpl, instruction in itertools.product(description_templates, instructions):
        d = apply_placeholders(tpl, instruction=instruction)
        baked_nums.setdefault(d, 0)
        baked_nums[d] += 1

    _baked_description_templates = list(baked_nums.keys())
    _baked_description_weights = np.fromiter(baked_nums.values(), float)
    _baked_description_weights /= np.sum(_baked_description_weights)

def choose_baked_description_template():
    bake_descriptions()
    return np.random.choice(_baked_description_templates, p=_baked_description_weights)

def generate_descriptions(n: int, shape: str, fill_color: str) -> List[str]:
    bake_descriptions()
    if n > len(_baked_description_templates):
        raise ValueError("cannot generate more than %d unique descriptions" % len(
            _baked_description_templates))

    chosen_templates = np.random.choice(
        _baked_description_templates, n, replace=False, p=_baked_description_weights)
    return [generate_single_description(shape, fill_color, tpl) for tpl in chosen_templates]


if __name__ == "__main__":
    n = 2
    shapes = ["square", "triangle", "star"]
    fill_colors = ["blue", "red", "green", "cyan"]

    np.random.seed(42)
    print("Single:", generate_single_description("square", "blue"))
    print("Multiple:")
    for shape in shapes:
        for fill_color in fill_colors:
            print("----")
            for d in sorted(generate_descriptions(n, shape, fill_color)):
                print(d)

    print()
    print("Total unique descriptions per configuration:",
          len(_baked_description_templates))
    print('\n----\n')

    print('Modifications:')
    for shape in shapes:
        print(generate_single_modification(new_shape=shape))
    print('----')
    for color in fill_colors:
        print(generate_single_modification(new_color=color))
