import os

import dill

from gauss import config
from gauss.domains.pandas_lite.definition import PandasLiteSynthesisDomain
from gauss.synthesis.config import EngineConfig
from gauss.synthesis.interface import build_synthesis_engine
from gauss.utilities.logutils import logger


def load_engine(name: str):
    path = f"{config.PROJECT_DIR}/.saved_engines/{name}/engine.pkl"
    if not os.path.exists(path):
        raise AssertionError(f"No engine found with name {name}.")

    with open(path, 'rb') as f:
        return dill.load(f)


def save_engine(engine, name: str):
    path = f"{config.PROJECT_DIR}/.saved_engines/{name}/engine.pkl"

    with open(path, 'wb') as f:
        dill.dump(engine, file=f)


def create_pandas_lite_engine(num_examples_per_component: int = 50,
                              max_length: int = 2,
                              max_inputs: int = 1):
    domain = PandasLiteSynthesisDomain()
    name = 'pandas_lite'
    cfg = EngineConfig(f"{config.PROJECT_DIR}/.saved_engines/{name}",
                       num_examples_per_component=num_examples_per_component,
                       max_length=max_length,
                       max_inputs=max_inputs,
                       overwrite=True)
    engine = build_synthesis_engine(domain, cfg)

    return engine


if __name__ == "__main__":
    create_pandas_lite_engine()
