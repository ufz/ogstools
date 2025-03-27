# ogstools/materiallib/validation.py

from ogstools.materiallib.schema.process_schema import PROCESS_SCHEMAS

from .core.medium import Medium


def validate_medium(medium: Medium, process: str) -> list[str]:
    schema = PROCESS_SCHEMAS.get(process)
    if schema is None:
        return [f"❌ No schema defined for process '{process}'"]

    required = set(schema.keys())
    provided = {p.name for p in medium.properties}

    missing = required - provided
    unused = provided - required

    messages = []

    if missing:
        messages.append(
            f"❌ Missing required properties: {', '.join(sorted(missing))}"
        )
    else:
        messages.append("✅ All required properties are present.")

    if unused:
        messages.append(
            f"⚠️  Unused properties (ignored for process '{process}'): {', '.join(sorted(unused))}"
        )

    return messages
