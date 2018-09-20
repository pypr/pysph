# We do this as we are not interested in the post-processing step.
if __name__ == '__main__':
    from pysph.examples.elliptical_drop import EllipticalDrop
    app = EllipticalDrop()
    app.run()
