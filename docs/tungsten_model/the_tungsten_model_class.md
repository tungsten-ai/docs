The ``TungstenModel`` class is the base class for all Tungsten model classes. A Tungsten model class looks something like this:
```python
from typing import List

import torch
from tungstenkit import io, model


class Input(io.BaseIO):
    ...


class Output(io.BaseIO):
    ...


@model.config(
    gpu=True,  # Whether to use a GPU or not
    batch_size=4,  # Max batch size for adaptive batching
    python_packages=["torch", "torchvision"],  # Required Python packages
    description="An example model"  # Model description
)
class Model(model.TungstenModel[Input, Output]):
    """
    A Tungsten model whose input/output types are 'Input' and 'Output', respectively.
    """
    def setup(self):
        """Load model weights"""
        weights = torch.load("weights.pth")
        self.model = load_model(weights)
    
    def predict(self, inputs: List[Input]) -> List[Output]:
        """Run a batch prediction"""
        inputs = preprocess(inputs)
        outputs = self.model(inputs)
        outputs = postprocess(outputs)
        return outputs
```

## Basic Usage
### Declare input/output types
Input/output types are declared by passing them as type arguments to ``TungstenModel`` class:

```python hl_lines="17"
from typing import List
import torch
from tungstenkit import io, model

class Input(io.BaseIO):
    ...

class Output(io.BaseIO):
    ...

@model.config(
    gpu=True,  # Whether to use a GPU or not
    batch_size=4,  # Max batch size for adaptive batching
    python_packages=["torch", "torchvision"],  # Required Python packages
    description="An example model"  # Model description
)
class Model(model.TungstenModel[Input, Output]):
    """
    A Tungsten model whose input/output types are 'Input' and 'Output', respectively.
    """
    def setup(self):
        """Load model weights"""
        weights = torch.load("weights.pth")
        self.model = load_model(weights)
    
    def predict(self, inputs: List[Input]) -> List[Output]:
        """Run a batch prediction"""
        inputs = preprocess(inputs)
        outputs = self.model(inputs)
        outputs = postprocess(outputs)
        return outputs
```

See [Tungsten Model - Input/Output](https://tungsten-ai.github.io/docs/tungsten_model/input_and_output) to learn how to define input/output.

### Define how to load a model
You can override the ``setup()`` method to define how to load a model:

```python hl_lines="25-28"
from typing import List

import torch
from tungstenkit import io, model


class Input(io.BaseIO):
    ...


class Output(io.BaseIO):
    ...


@model.config(
    gpu=True,  # Whether to use a GPU or not
    batch_size=4,  # Max batch size for adaptive batching
    python_packages=["torch", "torchvision"],  # Required Python packages
    description="An example model"  # Model description
)
class Model(model.TungstenModel[Input, Output]):
    """
    A Tungsten model whose input/output types are 'Input' and 'Output', respectively.
    """
    def setup(self):
        """Load model weights"""
        weights = torch.load("weights.pth")
        self.model = load_model(weights)
    
    def predict(self, inputs: List[Input]) -> List[Output]:
        """Run a batch prediction"""
        inputs = preprocess(inputs)
        outputs = self.model(inputs)
        outputs = postprocess(outputs)
        return outputs
```

As you can see, the ``weights.pth`` file is required to setup.
Before building, you should make sure that the file exists in the build directory.

### Define how a prediction works
The ``predict()`` method defines the computation performed at every prediction request.

It takes a non-empty list of ``Input`` objects as an argument, and should return a list of the same number of ``Output`` objects.

It should be overridden by all subclasses:

```python hl_lines="29-35"
from typing import List

import torch
from tungstenkit import io, model


class Input(io.BaseIO):
    ...


class Output(io.BaseIO):
    ...

@model.config(
    gpu=True,  # Whether to use a GPU or not
    batch_size=4,  # Max batch size for adaptive batching
    python_packages=["torch", "torchvision"],  # Required Python packages
    description="An example model"  # Model description
)
class Model(model.TungstenModel[Input, Output]):
    """
    A Tungsten model whose input/output types are 'Input' and 'Output', respectively.
    """
    def setup(self):
        """Load model weights"""
        weights = torch.load("weights.pth")
        self.model = load_model(weights)
    
    def predict(self, inputs: List[Input]) -> List[Output]:
        """Run a batch prediction"""
        inputs = preprocess(inputs)
        outputs = self.model(inputs)
        outputs = postprocess(outputs)
        return outputs
```

During runtime, Tungstenkit automatically keeps all options same in a batch.
So, you can define the ``predict`` method of a text-to-image generaton model like this:

```python hl_lines="6-13 20 22"
from typing import List
from tungstenkit import io

class Input(io.BaseIO):
    prompt: str
    width: int = io.Option(
        choices=[128, 256],
        default=256,
    )
    height: int = io.Option(
        choices=[128, 256],
        default=256,
    )

class Output(io.BaseIO):
    image: io.Image

class Model(model.TungstenModel[Input, Output]):
    def predict(self, inputs: List[Input]) -> List[Output]:
        options = inputs[0]
        prompts = [inp.prompt for inp in inputs]
        pil_images = self.model(prompts, width=options.width, height=options.height)
        images = [io.Image.from_pil_image(pil_image) for pil_image in pil_images]
        outputs = [Output(image=image) for image in images]
        return outputs
```

### Add dependencies and explanations
You can add dependencies and explanations via the ``config`` decorator:
```python hl_lines="15-20"
from typing import List

import torch
from tungstenkit import io, model


class Input(io.BaseIO):
    ...


class Output(io.BaseIO):
    ...


@model.config(
    gpu=True,  # Whether to use a GPU or not
    batch_size=4,  # Max batch size for adaptive batching
    python_packages=["torch", "torchvision"],  # Required Python packages
    description="An example model"  # Model description
)
class Model(model.TungstenModel[Input, Output]):
    """
    A Tungsten model whose input/output types are 'Input' and 'Output', respectively.
    """
    def setup(self):
        """Load model weights"""
        weights = torch.load("weights.pth")
        self.model = load_model(weights)
    
    def predict(self, inputs: List[Input]) -> List[Output]:
        """Run a batch prediction"""
        inputs = preprocess(inputs)
        outputs = self.model(inputs)
        outputs = postprocess(outputs)
        return outputs
```

The ``config`` decorator takes the following keyword-only arguments:

- ``description (str | None)``: A text explaining the model (default: ``None``).
- ``readme_md (str | None)``: Path to the ``README.md`` file (default: ``None``).
- ``batch_size (int)``: Max batch size for adaptive batching (default: ``1``).
- ``gpu (bool)``: Indicates if the model requires GPUs (default: ``False``).
- ``cuda_version (str | None)``: CUDA version in ``<major>[.<minor>[.<patch>]]`` format. If ``None`` (default), the cuda version will be automatically determined as compatible with ``python_packages``. Otherwise, fix the CUDA version as ``cuda_version``.
- ``gpu_mem_gb (int)``: Minimum GPU memory size required to run the model (default: ``16``). This argument will be ignored if ``gpu==False``.
- ``python_packages (list[str] | None)``: A list of pip requirements in ``<name>[==<version>]`` format. If ``None`` (default), no extra Python packages are added.
- ``python_version (str | None)``: Python version to use in ``<major>[.<minor>]`` format. If ``None`` (default), the python version will be automatically determined as compatible with ``python_packages`` while prefering the current Python version. Otherwise, fix the Python version as ``python_version``.
- ``system_packages (list[str] | None)``: A list of system packages that will installed by the system package manager (e.g. ``apt``). The default value is ``None``. This argument will be ignored while using a custom base image, because Tungstenkit cannot decide which package manager to use.
- ``mem_gb (int)``: Minimum memory size required to run the model (default: ``8``).
- ``include_files (list[str] | None)``: A list of patterns as in ``.gitignore``. If ``None`` (default), all files in the working directory and its subdirectories are added, which is equivalent to ``[*]``.
- ``exclude_files (list[str] | None)``: A list of patterns as in ``.gitignore`` for matching which files to exclude. If ``None`` (default), all hidden files and Python bytecodes are ignored, which is equivalent to ``[".*/", "__pycache__/", "*.pyc", "*.pyo", "*.pyd"]``.
- ``dockerfile_commands (list[str] | None)``: A list of dockerfile commands (default: ``None``). The commands will be executed before setting up python packages.
- ``base_image (str | None)``: Base docker image in ``<repository>[:<tag>]`` format. If ``None`` (default), the base image is automatically selected with respect to pip packages, the device type, and the CUDA version. Otherwise, use it as the base image and ``system_packages`` will be ignored.

## Advanced Usage
### Define how a demo prediction works
You can define an object detection model like this:

```python
from typing import List
from tungstenkit import io

class BoundingBox(io.BaseIO):
    xmin: int
    xmax: int
    ymin: int
    ymax: int

class Detection(io.BaseIO):
    label: str
    bbox: BoundingBox

class Input(io.BaseIO):
    image: io.Image

class Output(io.BaseIO):
    detections: List[Detection]

class Model(model.TungstenModel[Input, Output]):
    def setup(self):
        ...

    def predict(self, inputs: List[Input]) -> List[Output]:
        ...
```

It looks fine at first glance, but there is a problem in the demo page.
All that users can get is just a raw JSON.

Then, you could add the visualization result to the output:

```python hl_lines="19"
from typing import List
from tungstenkit import io

class BoundingBox(io.BaseIO):
    xmin: int
    xmax: int
    ymin: int
    ymax: int

class Detection(io.BaseIO):
    label: str
    bbox: BoundingBox

class Input(io.BaseIO):
    image: io.Image

class Output(io.BaseIO):
    detections: List[Detection]
    visualized: io.Image

class Model(model.TungstenModel[Input, Output]):
    def setup(self):
        ...

    def predict(self, inputs: List[Input]) -> List[Output]:
        ...
```

This may satisfy demo users, but API users will suffer performance degradation due to visualization overhead.

In such a case, you can separate the method for demo predictions:

```python hl_lines="20-21 30-38"
from typing import List, Tuple, Dict
from tungstenkit import io

class BoundingBox(io.BaseIO):
    xmin: int
    xmax: int
    ymin: int
    ymax: int

class Detection(io.BaseIO):
    label: str
    bbox: BoundingBox

class Input(io.BaseIO):
    image: io.Image

class Output(io.BaseIO):
    detections: List[Detection]

class Visualization(io.BaseIO):
    result: io.Image

class Model(model.TungstenModel[Input, Output]):
    def setup(self):
        ...

    def predict(self, inputs: List[Input]) -> List[Output]:
        ...
    
    def predict_demo(
        self, 
        inputs: List[Input]
    ) -> Tuple[List[Output], List[Visualization]]:
        outputs = self.predict(inputs)
        pil_images = visualize(inputs, outputs)
        images = [io.Image.from_pil_image(pil_image) for pil_image in pil_images]
        visualizations = [Visualization(result=image) for image in images]
        return outputs, visualizations
```

Then, the ``predict`` method is executed when a prediction is requested through the API. 
On the other hand, for a demo request, the ``predict_demo`` method is executed, and the visualization result is shown on the demo page.