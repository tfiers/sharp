def run():
    from preload import preload

    preload(["scipy.signal", "fklab.segments", "matplotlib.pyplot"])

    from sharp.init import sharp_workflow
    from sharp.workflow import compose_workflow

    compose_workflow()
    sharp_workflow.run()


if __name__ == "__main__":
    run()
