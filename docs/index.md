<p align="center">
  <a href="https://tungsten-ai.github.io/tungsten-docs"><img src="images/logo.svg" alt="Tungsten" width="50%" height="50%"></a>
</p>
<p align="center">
  <img src="images/demo.gif" alt="Tungsten">
</p>

---
Tungsten is the easiest way to share and manage ML models.

🚀 **Build once, use everywhere**  
Tungsten-built ML model containers can be used as RESTful API servers, GUI/CLI applications, serverless functions, and functions in Python scripts without any model-specific setup.

⚙️ **Manage all in one place**  
Tungsten stores every version of ML models, data, and test specs. Also, it automatically runs tests and keeps evaluation scores up-to-date.
So, users can easily run, compare, and download ML models.

## Key Features
- [Build only with a few lines of Python codes](#build-only-with-a-few-lines-of-python-codes)
- [Automatically generate a standardized RESTful API for a model](#automatically-generate-a-standardized-restful-api-for-a-model)
- [Provide a clean and intuitive web UI for a model](#provide-a-clean-and-intuitive-web-ui-for-a-model)
- [Allow your own machines to be used to run models](#allow-your-own-machines-to-be-used-to-run-models)
- Model, test data, and test spec versioning (comming soon)
- Keep test scores up-to-date (comming soon)

For a complete example including more features, see the [Tutorial - Getting Started](https://tungsten-ai.github.io/tungsten-docs/tutorial/getting_started/).

---

## Requirements
- Python >= 3.7
- [Docker](https://docs.docker.com/engine/install/)
- (Optional) [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) for running GPU models locally. But you can build, push, run remotely a GPU model without it.

## Installation
```
pip install tungstenkit
```

---

## Take the tour
### Build only with a few lines of Python codes
Tungsten does not require any complex configuration file for building. 

All you have to do is write a simple ``tungsten_model.py`` like below:
```python
from typing import List, Tuple

import torch
from tungstenkit.io import BaseIO, Image
from tungstenkit.model import TungstenModel, config


class Input(BaseIO):
    image: Image


class Output(BaseIO):
    score: float
    label: str


@config(
    gpu=True,
    python_packages=["torch", "torchvision"],
    batch_size=64,
    description="A torch model"
)
class Model(TungstenModel[Input, Output]):
    def setup(self):
        self.model = torch.load("./weights.pth")

    def predict(self, inputs: List[Input]) -> List[Output]:
        input_tensor = preprocess(inputs)
        output_tensor = self.model(input_tensor)
        outputs = postprocess(output_tensor)
        return outputs
```
Now you can start building:
```console
$ tungsten build -n tungsten-example

✅ Successfully built tungsten model: 'tungsten-basic:latest'
```

### Automatically generate a standardized RESTful API for a model

The model container is a standardized RESTful API server itself.

Run the container:

```console
$ docker run -p 3000:3000 tungsten-example:latest

INFO:     Setting up the model
INFO:     Getting inputs from the input queue
INFO:     Starting the prediction service
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:3000 (Press CTRL+C to quit)
```

Now you can run predictions using the server. For example,
```console
$ curl -X 'POST' 'http://localhost:3000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '[{"image": "https://picsum.photos/200.jpg"}]'

{
    "status": "success",
    "outputs": [{"score": 0.5, "label": "dog"}],
    "error_message": null
}
```

Also, a Swagger documentation for the server is automatically generated.

Visit [http://localhost:3000/docs](http://localhost:3000/docs) in a browser:

![tungsten-model-api](images/model-api.png "Tungsten Model API")

### Provide a clean and intuitive web UI for a model
#### Option 1. Run remotely
First, login to a Tungsten server:
```
$ tungsten login

Login Success!
```
Then, push a model:
```
$ tungsten push <user name>/<project name> -n tungsten-example:latest

✅ Successfully pushed to 'https://server.tungsten-ai.com'
```

Now you can run this model in the Tungsten dashboard.

Visit [https://tungsten-ai.com](https://tungsten-ai.com) in a browser:

![tungsten-dashboard](images/demo.gif "Tungsten Dashboard")

#### Option 2. Run locally
You can run a GUI app locally in a single command:
```
$ tungsten demo tungsten-example:latest -p 8080

INFO:     Uvicorn running on http://localhost:8080 (Press CTRL+C to quit)
```

Visit [http://localhost:8080](http://localhost:8080) in a browser:

![tungsten-dashboard](images/demo.gif "Tungsten Dashboard")


### Allow your own machines to be used to run models
You can register Tungsten runners to a Tungsten server and make the server use your own machines for running models.

Register a runner:

```console
$ tungsten-runner register

Enter runner mode (pipeline, prediction) [prediction]: prediction
Enter URL of the tungsten server: https://server.tungsten-ai.com
Enter registration token: C6r5rp2PhfdXbJtFbBMhifgLDhagAc
Enter runner name [mydesktop]: myrunner 
Enter tags (comma separated) []: myrunnergroup
Enter GPU index to use []: 0
Runner 'mjpyeon-desktop' is registered - id: 245
Updated runner config
```

Run all registered runners:

```console
$ tungsten-runner run

Runner 0   | running  2023-04-21 16:59:14.490 | INFO     | Fetching a prediction job
                      2023-04-21 16:59:49.184 | INFO     | Job 0f7c50867417456ebd1389cfb74e489f assigned
Runner 1   | running  2023-04-21 16:59:14.490 | INFO     | Fetching a prediction job
```


## License
This project is licensed under the terms of the Apache License 2.0.
