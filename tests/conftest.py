from hypothesis import Verbosity, settings

settings.register_profile("ci", max_examples=250, deadline=1000)
settings.register_profile("default", max_examples=50, deadline=350)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose)

settings.load_profile("default")


def pytest_make_parametrize_id(config, val, argname):  # noqa: ARG001
    if hasattr(val, "output_name"):
        return val.output_name
    if isinstance(val, dict):
        values = "_".join(f"{k}_{v}" for k, v in val.items())
        if argname == "kwargs":
            return values
        return f"{argname}_{values}"
    if callable(val):
        return "lambda"
    return None  # lets pytest handle the rest
