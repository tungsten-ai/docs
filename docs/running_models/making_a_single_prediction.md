After [building](https://tungsten-ai.github.io/docs/building_your_model/containerizing/) or [pulling](https://tungsten-ai.github.io/docs/pushing_and_pulling_models/pulling/) a model, you can make a prediction with it.

## Using Tungstenkit CLI

```console
$ tungsten predict text-to-image:v1 \
   -i prompt="a professional photograph of an astronaut riding a horse" \
   -i seed=1234

{
  "image": "./output.png"
}
```


If you want to run with the latest model in a repository, you can omit the tag:
```console
$ tungsten predict text-to-image \
   -i prompt="a professional photograph of an astronaut riding a horse" \
   -i seed=1234
```

If you want to run with the latest model, you can omit the model name.
```console
$ tungsten predict \
   -i prompt="a professional photograph of an astronaut riding a horse" \
   -i seed=1234
```


### Options
  - ``--input (-i)``: Input field in the format of ``<NAME>=<VALUE>``
  - ``--output-file-dir``: Output file directory
    - Default: ``.`


## Using Python
```python
>>> from tungstenkit import models
>>> model = models.get("text-to-image")
>>> model.predict(
    {"prompt": "a professional photograph of an astronaut riding a horse"}
)
{"image": PosixPath("./output.png")}
```

You can also make a prediction with multiple inputs.
```python
>>> model.predict(
    [{"prompt": "astronaut"}, {"prompt": "horse"}]
)
[{"image": PosixPath("./output-0.png")}, {"image": PosixPath("./output-1.png")}]
```