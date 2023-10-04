## ``GET /``
Get input/output JSON schemas of this model. 

The response is a JSON object with the following fields:

- ``input_schema``: Input JSON schema. Keys are same as [the input class declared using ``define_model`` decorator](https://tungsten-ai.github.io/docs/building_your_model/model_definition).
- ``output_schema``: Output JSON schema. Keys are same as [the output class declared using ``define_model`` decorator](https://tungsten-ai.github.io/docs/building_your_model/model_definition). 
- ``demo_output_schema``: Demo output JSON schema. Keys are same as [the demo output class declared using ``define_model`` decorator](https://tungsten-ai.github.io/docs/building_your_model/model_definition).

<!-- 

``POST /predict``
Execute a **synchronous** prediction. 

The request body should be a list of input JSON objects.  

The response is a JSON object with the following fields:

- ``outputs``: List of output JSON objects. 
- ``status``: Either ``success`` or ``failed``.
- ``error_message``: The error message if ``status`` if ``failed``.


For example, if you define input/output as the following:

```python
from tungstenkit import BaseIO, define_model, Image


class Input(BaseIO):
    prompt: str
    seed: int


class Output(BaseIO):
    image: Image


@define_model(input=Input, output=Output)
class TextToImageModel:
    ...
```

a response body could be:
```json
[
    {
        "prompt": "astronaut",
        "seed": 1234
    }, 
    {
        "prompt": "cat",
        "seed": 5678
    }

]
```

and the response body would be:
```json
{
    "outputs": [
        {
            "image": "data:image/png;base64,..."
        },
        {
            "image": "data:image/png;base64,..."
        }

    ]
}
``` 
 -->


## ``POST /predictions``
Create a prediction. 

The request body should be a list of input JSON objects.  

The response is a JSON object with the ``prediction_id`` field.

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


## ``GET /predictions/{prediction_id}``
Get the result of an asynchronous prediction.

The response is a JSON object with the following fields:

- ``outputs``: List of output JSON objects. 
- ``status``: Among ``pending``, ``running``, ``success`` and ``failed``.
- ``error_message``: The error message if ``status`` is ``failed``.

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

## ``POST /predictions/{prediction_id}/cancel``
Cancel an asynchronous prediction.


## ``POST /demo``
Make a demo prediction. 

The request body should be a list of input JSON objects.  

The response is a JSON object with the ``demo_id`` field.


## ``GET /demo/{demo_id}``
Get the result of a demo prediction.

The response is a JSON object with the following fields:

- ``outputs``: List of output JSON objects. 
- ``demo_outputs``: List of demo output JSON objects.
- ``status``: Among ``pending``, ``running``, ``success`` and ``failed``.
- ``error_message``: The error message if ``status`` if ``failed``.
- ``logs``: Logs while running [the ``predict_demo`` method](https://tungsten-ai.github.io/docs/building_your_model/model_definition).

## ``POST /demo/{demo_id}/cancel``
Cancel a demo prediction.

