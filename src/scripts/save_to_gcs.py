import pickle

def save_to_gcs(payload: dict,
                uri: str) -> int:
    """
    :param payload: Dictionary of payload to save to GCS
    :param uri: Cloud Storage URI
    """
    with open(uri) as file:
        file.writelines(payload)

    return 0
