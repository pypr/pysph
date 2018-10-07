# We do this as we are not interested in the post-processing step.
if __name__ == '__main__':
    from pysph.examples.cavity import LidDrivenCavity
    app = LidDrivenCavity()
    app.run()
