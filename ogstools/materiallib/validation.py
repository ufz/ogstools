# ogstools/materiallib/validation.py

from ogstools.materiallib.schema.process_schema import PROCESS_SCHEMAS

def validate_medium(medium) -> list[str]:
    schema = PROCESS_SCHEMAS.get(medium.process)
    if schema is None:
        return [f"❌ No schema defined for process '{medium.process}'"]

    required = set(schema.keys())
    provided = set([p.name for p in medium.properties])

    missing = required - provided
    unused = provided - required

    messages = []

    if missing:
        messages.append(f"❌ Missing required properties: {', '.join(sorted(missing))}")
    else:
        messages.append("✅ All required properties are present.")

    if unused:
        messages.append(f"⚠️  Unused properties (ignored for process '{medium.process}'): {', '.join(sorted(unused))}")

    return messages