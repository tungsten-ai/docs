## Global options
  - ``--debug``: Show logs for debugging.
  - ``--help (-h)``: Show this message and exit.

## build
This command is for containerizing your model.

```
tungsten build .
```
will create a docker image with ``tungsten_model.py`` in the current directory.

### Options
  - ``--name (-n)``: Name of the model in ``<repo name>[:<tag>]`` format
  - ``--model-module (-m)``: Model module (e.g., ``some.example.module``)
    - default: ``tungsten_model``
  - ``--model-class (-c)``: Model class (e.g., ``MyModel``)  
    - default: the class decorated with ``tungstenkit.define_model``
  - ``--copy-files``: Copy files to the container in ``<src in host>:<dest in container>`` format

## models

## serve

## demo

## predict

## tag

## remove

## clear

## push

## pull