## Global options
  - ``--debug``: Show logs for debugging.
  - ``--help (-h)``: Show the help message and exit.

## ``build``
This command is for containerizing your model.

```bash
tungsten build .
```
will create a docker image with ``tungsten_model.py`` in the current directory.

### Options
  - ``--name (-n)``: Name of the model in ``<repo name>[:<tag>]`` format
    - Default: ``BUILD_DIRECTORY_NAME:MODEL_ID`` (e.g. ``current-directory:41ea3bf``)
  - ``--model-module (-m)``: Model module (e.g., ``some.example.module``)
    - Default: ``tungsten_model``
  - ``--model-class (-c)``: Model class (e.g., ``MyModel``)  
    - Default: the class decorated with ``tungstenkit.define_model``
  - ``--copy-files``: Copy files to the container in ``<src in host>:<dest in container>`` format

## ``models``
The ``models`` command displays all the available models.

```bash
tungsten models
```

## ``serve``
The ``serve`` command runs a REST API server with a model.
```bash
tungsten serve mymodel:v1
```

If you want to run with the latest model in a repository, you can omit the tag.
```bash
tungsten serve mymodel
```

If you want to run with the latest model, you can omit the model name.
```bash
tungsten serve
```

### Options
  - ``--port (-p)``: Bind socket to this port.
    - Default: ``3000``
  - ``--batch-size``: Max batch size for adaptive batching.
    - Default: Declared value in ``tungsten_model.py``
  - ``--log-level``: Log level of the server.
    - Default: ``info``
    - Available values: ``trace``, ``debug``, ``info``, ``warning``, ``error``


## ``demo``
The ``demo`` command runs an interactive web demo with a model.
```bash
tungsten demo mymodel:v1
```

If you want to run with the latest model in a repository, you can omit the tag.
```bash
tungsten demo mymodel
```

If you want to run with the latest model, you can omit the model name.
```bash
tungsten demo
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

## ``predict``
This command is for running a prediction with a model.

```bash
tungsten predict stable-diffusion:1.5 -i prompt="astronaut" -i seed=1234
```

### Options
  - ``--input (-i)``: Input field in the format of ``<NAME>=<VALUE>``
  - ``--output-file-dir``: Output file directory
    - Default: ``.``

## ``tag``
The ``tag`` command adds a new name of a model.

```bash
tungsten tag mymodel:v1 mymodel:v1.1
```

## ``remove``
The ``remove`` command removes a model.

```bash
tungsten remove mymodel:v1
```

## ``clear``
The ``clear`` command removes all models.

```bash
tungsten clear
```

If you want to remove all models in a repository (e.g. all models whose name starts with ``mymodel``), you can put the repository name.

```bash
tungsten clear mymodel
```

## ``login``
This command is for logging in to [tungsten.run](https://tungsten.run).

```bash
tungsten login
```

## ``push``
This command is for pushing a model to [tungsten.run](https://tungsten.run).
Before running this command, you should login to [tungsten.run](https://tungsten.run) using [``login``](#login) command.

```bash
tungsten push exampleuser/exampleproject:exampleversion
```

If you already logged in with the username of ``exampleuser``, the above command is equivalent to:

```bash
tungsten push exampleproject:exampleversion
```

If you want to push the latest model, you can omit the model name.
```bash
tungsten push
```

## ``pull``
This command is for pulling a model from [tungsten.run](https://tungsten.run).

```bash
tungsten pull exampleuser/exampleproject:exampleversion
```