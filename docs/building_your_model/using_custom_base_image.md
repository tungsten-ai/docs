By default, Tungstenkit uses a Python or CUDA image as the base image for containerization. But, you can manually set a base image. 

You can use a custom base image by setting ``base_image`` in the ``define_model`` decorator:
```python hl_lines="7"
@define_model(
    input=SDInput,
    output=SDOutput,
    batch_size=1,
    gpu=True,
    gpu_mem_gb=14,
    base_image="mjpyeon/tungsten-sd-base:v1",
)
class StableDiffusion:
    ...
```

In this case, since Tungstenkit doesn't know which OS the base image contains, ``system_packages`` argument of ``define_model`` decorator will be ignored.

