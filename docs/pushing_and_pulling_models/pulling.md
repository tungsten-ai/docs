You can pull and use models from [tungsten.run](https://tungsten.run).

```console
$ tungsten pull exampleuser/exampleproject:v1

✅ Successfully pulled 'exampleuser/exampleproject:v1' from 'https://api.tungsten.run'
```

Then you can see that this model has been added to the model list:
```console
$ tungsten models

Repository                     Tag   Create Time          Docker Image ID
-----------------------------  ---   -------------------  ---------------
exampleuser/exampleproject:v1  v1    2023-04-26 05:23:58  830eb82f0fcd
```