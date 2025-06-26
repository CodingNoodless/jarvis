import importlib
import traceback

def dispatch(intent_obj, context=None, memory=None):
    intent = intent_obj.get("intent", "unknown")
    try:
        module = importlib.import_module(f"skills.{intent}")
        return module.run(intent_obj, context or {}, memory or {})
    except ModuleNotFoundError:
        try:
            module = importlib.import_module("skills.fallback")
        except Exception:
            return "Fallback skill is missing. Please create 'skills/fallback.py'."
    try:
        return module.run(intent_obj, context or {}, memory or {})
    except Exception as e:
        traceback.print_exc()
        return f"There was an error executing the skill for '{intent}': {e}"
