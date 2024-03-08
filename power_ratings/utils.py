import typing as T


def flatten_params_for_logging(params: T.Dict[str, T.Any]) -> T.Dict[str, str]:
    """
    Since mlflow wants only str as params, collaps Dict objects
    """
    output = {}
    for k, v in params.items():
        if isinstance(v, dict):
            flattened = flatten_params_for_logging(v)
            for k2, v2 in flattened.items():
                output[f"{k}.{k2}"] = v2
        elif isinstance(v, str):
            output[k] = v[:250]
        elif isinstance(v, list):
            output[k] = f"[{', '.join(v)}]"[:250]
        else:
            output[k] = str(v)[:250]
    return output
