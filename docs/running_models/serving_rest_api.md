After [building](https://tungsten-ai.github.io/docs/building_your_model/containerizing/) or [pulling](https://tungsten-ai.github.io/docs/pushing_and_pulling_models/pulling/) a model, you can serve a REST API with it.


## Starting a server
### Using Tungstenkit CLI

```console
$ tungsten serve text-to-image:v1 -p 3000

INFO:     Uvicorn running on http://0.0.0.0:3000 (Press CTRL+C to quit)
```

If you want to run with the latest model in a repository, you can omit the tag:
```console
$ tungsten serve text-to-image
```

If you want to run with the latest model, you can omit the model name.
```console
$ tungsten serve
```

#### Options
  - ``--port (-p)``: Bind socket to this port.
    - Default: ``3000``
  - ``--batch-size``: Max batch size for adaptive batching.
    - Default: Declared value using the ``define_model`` decorator
  - ``--log-level``: Log level of the server.
    - Default: ``info``
    - Available values: ``trace``, ``debug``, ``info``, ``warning``, ``error``


### Using Docker
```console
$ docker run --gpus 0 --rm -p 3000:3000 text-to-image:v1

INFO:     Uvicorn running on http://0.0.0.0:3000 (Press CTRL+C to quit)
```

Refer to [Docker docs](https://docs.docker.com/engine/reference/commandline/run/) for more details on the ``docker run`` command.

## Running predictions

### Starting a prediction
You can make a prediction with multiple inputs by sending a request to ``POST /predictions`` endpoint.

The request body should be a list of input JSON objects. Keys of input JSON objects should be same as field names of [the input class declared using ``define_model`` decorator](https://tungsten-ai.github.io/docs/building_your_model/model_definition).

The response body has the ``prediction_id`` field.

For example, let's assume that we defined the input as follows:

```python
from tungstenkit import BaseIO, define_model


class Input(BaseIO):
    prompt: str
    seed: int


@define_model(input=Input, ...)
class TextToImageModel:
    ...
```

Then, you can send a request as follows:

```console
$ curl -X 'POST' 'http://localhost:3000/predictions' \
  -H 'Accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '[{"prompt": "a professional photograph of an astronaut riding a horse"}]'

{
    "prediction_id": "39c9eb6b"
}
```

### Getting the status and result of a prediction
``GET /predictions/{prediction_id}`` endpoint provides the status and result of a prediction.

The response is a JSON object with the following fields:

- ``outputs``: List of output JSON objects. 
- ``status``: Among ``pending``, ``running``, ``success`` and ``failed``.
- ``error_message``: The error message if ``status`` if ``failed``.

For example, let's assume that we defined the output as follows:

```python
from tungstenkit import BaseIO, define_model, Image


class Output(BaseIO):
    image: Image


@define_model(output=Output, ...)
class TextToImageModel:
    ...
```

Then, you can send a request as follows:


```console
$ curl -X 'GET' 'http://localhost:3000/predictions/39c9eb6b' \
  -H 'Accept: application/json'

{
    "outputs": [{"image": "data:image/png;base64,..."}],
    "status": "success"
}
```

### Canceling a prediction
You can cancel a prediction via ``POST /predictions/{prediction_id}/cancel`` endpoint.

```console
$ curl -X 'GET' 'http://localhost:3000/predictions/39c9eb6b/cancel'
```

### Further information
See [REST API reference](https://tungsten-ai.github.io/docs/rest_api_reference) for more details and other endpoints.
