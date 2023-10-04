With ``tungsten_model.py``, you can containerize a model using ``tungsten build`` command:

```console
$ tungsten build . -n mymodel

✅ Successfully built tungsten model: 'mymodel:e3a5de56'
```

Then you can see that this model has been added to the model list:
```console
$ tungsten models

Repository        Tag       Create Time          Docker Image ID
----------------  --------  -------------------  ---------------
mymodel           e3a5de56    2023-04-26 05:23:58  830eb82f0fcd
```