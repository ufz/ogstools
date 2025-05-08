from hypothesis import Verbosity, settings

settings.register_profile("ci", max_examples=1000, deadline=10000)
settings.register_profile("default", max_examples=100, deadline=350)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose)

settings.load_profile("default")
