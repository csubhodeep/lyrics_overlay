from functools import wraps
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple


def make_immutable(
    allowed_settable_attributes: Optional[Tuple[str, ...]] = None
) -> Callable:
    def _make_immutable(function: Callable) -> Callable:
        @wraps(function)
        def wrapper(*args, **kwargs: Dict[str, Any]) -> Callable:
            instance = args[0]
            prop_name = args[1]
            if hasattr(instance, prop_name):
                raise Exception(f"Value for attribute {prop_name} is already set")
            else:
                if allowed_settable_attributes:
                    if prop_name in allowed_settable_attributes:
                        return function(*args, **kwargs)
                    else:
                        raise Exception(
                            f"{prop_name} is not a valid attribute for {instance.__class__.__name__}"
                        )
                else:
                    return function(*args, **kwargs)

        return wrapper

    return _make_immutable
