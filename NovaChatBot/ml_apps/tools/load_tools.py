import numpy as np
import os
import json


def load_Json(file):
    with open(file) as js_file:
        data = json.load(js_file)

    return data