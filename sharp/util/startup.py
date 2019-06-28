from importlib import import_module


def preload(
    heavy_modules=("scipy.signal", "fklab.segments", "matplotlib.pyplot")
):
    """ Give feedback to user while program hangs at start. """
    print("Importing:")
    for mod in heavy_modules:
        print(f" - {mod}..", end=" ", flush=True)
        import_module(mod)
        print("✓")
