You can define input and output for image classification like this:

```python
from tungstenkit import io

class Input(io.BaseIO):
    image: io.Image

class Output(io.BaseIO):
    score: float
    label: str
```

Here you see:

- Input and output classes are inherited from [``tungstenkit.io.BaseIO``](#the-baseio-class).
- Input has a field named 'image', and its type is [``tungstenkit.io.Image``](#files).
- Output has two fields named 'score' and 'label', and their types are ``float`` and ``str``.

## The ``BaseIO`` class
The ``BaseIO`` class is the base class for all inputs and outputs in Tungstenkit. 

It is a simple wrapper of the [``BaseModel``](https://docs.pydantic.dev/latest/usage/models/) class in [Pydantic](https://docs.pydantic.dev/latest/), so you can use [all useful methods and attributes it provides](https://docs.pydantic.dev/latest/usage/models/#model-properties).


## Supported field types
Tungstenkit currently supports the following input/output field types:

| Type                                       | Input   | Output  |
| ------------------------------------------ | ------- | ------- |
| ``tungstenkit.io.Image``                   |    ✅   |    ✅    |
| ``tungstenkit.io.Video``                   |    ✅   |    ✅    |
| ``tungstenkit.io.Audio``                   |    ✅   |    ✅    |
| ``tungstenkit.io.Binary``                  |    ✅   |    ✅    |
| ``str``                                    |    ✅   |    ✅    |
| ``float``                                  |    ✅   |    ✅    |
| ``int``                                    |    ✅   |    ✅    |
| ``bool``                                   |    ✅   |    ✅    |
| ``dict`` or ``typing.Dict``                |    ❌   |    ✅    |
| ``list`` or ``typing.List``                |    ❌   |    ✅    |
| ``tuple`` or ``typing.Tuple``              |    ❌   |    ✅    |
| A subclass of ``tungstenkit.io.BaseIO``    |    ❌   |    ✅    |

For ``dict``, ``list``, ``tuple``, ``typing.Dict``, ``typing.List`` and ``typing.Tuple``, type arguments are required. For example, you should use ``dict[str, str]`` instead of ``dict``.

## Files
The ``tungstenkit.io`` module provides four primitives for files: ``Image``, ``Video``, ``Audio``, and ``Binary``.
They possess the following property and method:

- ``path`` : a string of the file path.
- ``from_path(path: StrPath)``: a class method for creating file objects from a filepath.  
```python
>>> from tungstenkit import io
>>> video_path = "video.mp4"
>>> video = io.Video.from_path(video_path)
>>> video.path
'/home/tungsten/working_dir/video.mp4'
```

The ``Image`` object has more methods:

- ``from_pil_image(pil_image: PIL.Image.Image)``: a class method for creating the object from a [``PIL.Image.Image``](https://pillow.readthedocs.io/en/stable/reference/Image.html#the-image-class) object.
- ``to_pil_image(mode: str = "RGB")``: returns a [``PIL.Image.Image``](https://pillow.readthedocs.io/en/stable/reference/Image.html#the-image-class) object.


## Input field descriptors
The ``tungstenkit.io`` module contains two input field descriptors: ``Field`` and ``Option`` functions:

- ``Field``: For setting properties of a *required* field.
- ``Option``: For declaring a field as *optional* and setting its properties. Optional fields will be same in an input batch and hidden in the model demo page by default.

Using them, you can:

- Distinguish between required and optional fields.
- Restrict input field values.
- Set input field descriptions shown in the model demo page.

For example, you can define an input class for text-to-image generation as follows:
```python
from tungstenkit import io

class Input(io.BaseIO):
    prompt: str = io.Field(
        description="Input prompt", 
        min_length=1, 
        max_length=200
    )
    width: int = io.Option(
        description="Width of output image",
        choices=[128, 256, 512],
        default=512,
    )
    height: int = io.Option(
        description="Height of output image",
        choices=[128, 256, 512],
        default=512,
    )
```

Both field descriptors take the following keyword-only arguments:

- ``description`` (``str``, optional): Human-readable description.
- ``ge`` (``float``, optional): Greater than or equal. If set, value must be greater than or equal to this. Only applicable to ``int`` and ``float``.
- ``le`` (``float``, optional): Less than or equal. If set, value must be less than or equal to this. Only applicable to ``int`` and ``float``.
- ``min_length`` (``int``, optional): Minimum length for strings.
- ``max_length`` (``int``, optional): Maximum length for strings.
- ``choices`` (``list``, optional): List of possible values. If set, value must be among a value in this.

The ``Option`` function additionally takes a positional argument:

- ``default`` (``Any``, required): Default value.