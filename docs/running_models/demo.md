After [building](https://tungsten-ai.github.io/docs/building_your_model/containerizing/) or [pulling](https://tungsten-ai.github.io/docs/pushing_and_pulling_models/pulling/) a model, you can start an interactive web demo with it.

Start the demo:
```console
$ tungsten demo text-to-image:v1 -p 8080

INFO:     Uvicorn running on http://localhost:8080 (Press CTRL+C to quit)
```

Visit [http://localhost:8080](http://localhost:8080):
![tungsten-dashboard](https://github.com/tungsten-ai/assets/blob/main/common/local-model-demo.gif?raw=true "Demo GIF")

If you want to run with the latest model in a repository, you can omit the tag:
```
$ tungsten demo text-to-image
```

If you want to run with the latest model, you can omit the model name.
```
$ tungsten demo
```

### Options
  - ``--port (-p)``: The port on which the demo server will listen.
    - Default: ``3300``
  - ``--host``: The host on which the demo server will listen
    - Default: ``localhost``
  - ``--batch-size``: Max batch size for adaptive batching.
    - Default: Declared value in ``tungsten_model.py``
  - ``--log-level``: Log level of the server.
    - Default: ``info``
    - Available values: ``trace``, ``debug``, ``info``, ``warning``, ``error``


