You can store, share and run your models in [tungsten.run](https://tungsten.run).

First, [create a project in tungsten.run](https://tungsten.run/new) in a browser.

Then, log in using Tungstenkit CLI to fetch the credential.

```console
$ tungsten login

User (username or email): exampleuser
Password:
Login Success!
```

Tag a model as ``<USER_NAME>/<PROJECT_NAME>:<VERSION>`` format:
```console
$ tungsten tag mymodel:v1 exampleuser/exampleproject:v1

Tagged model 'mymodel:v1' to 'exampleuser/exampleproject:v1'.
```

Push it:
```console
$ tungsten push exampleuser/exampleproject:v1

✅ Successfully pushed 'exampleuser/exampleproject:v1' to 'https://api.tungsten.run'
  - project: exampleuser/exampleproject
  - version: v1
```