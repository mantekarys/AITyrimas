import time
from typing import Any, Callable, Dict, Tuple, Union

import mlflow
import numpy as np

# import cv2
import torch
import torchvision
from metadrive import MetaDriveEnv, SafeMetaDriveEnv
from stable_baselines3.common.logger import KVWriter
from torchvision import transforms


def resize(image: np.ndarray):
    transformation = transforms.Resize(size=(84, 84))
    image: torch.Tensor = torch.tensor(image)
    shape = image.shape
    concatenated = image.reshape(shape=tuple([*shape[:-2], shape[-2] * shape[-1]]))
    chanel_second = concatenated.permute((0, 3, 1, 2))
    resized_image: torch.Tensor = transformation(chanel_second)
    chanel_last = resized_image.permute((0, 2, 3, 1))
    primary_shape = chanel_last.reshape(shape=tuple([shape[0], 84, 84, 3, 4]))
    return primary_shape


class FixedSafeMetaDriveEnv(SafeMetaDriveEnv):
    def reset(self, *args, **kwargs):
        return super().reset()


def linear_decay_schedule(lr_start: float) -> Callable[[float], float]:
    def schedule(progress_left: float) -> float:
        return lr_start * progress_left

    return schedule


def flat_dict(nested_dict: dict, prefix: str = "") -> dict:
    flat_level = {}
    for k, v in nested_dict.items():
        new_key_name = f"{prefix}/{k}".strip("/")
        if isinstance(v, dict):
            lower_level = flat_dict(v, prefix=new_key_name)
            flat_level.update(**lower_level)
        else:
            flat_level[new_key_name] = str(v)
    return flat_level


def measure_time(func: Callable, *args, **kwargs) -> tuple[float, Any]:
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return (end - start, result)


class MLflowOutputFormat(KVWriter):
    """
    Dumps key/value pairs into MLflow's numeric format.
    """

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:

        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):

            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)
