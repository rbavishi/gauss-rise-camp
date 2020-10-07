import uuid
from typing import List, Any, Dict, Type, Tuple

import dill
from IPython.display import display
from ipywidgets import widgets

from gauss import config
from gauss.engines import load_engine
from gauss.domains.pandas_lite.definition import PandasLiteSynthesisDomain, PandasLiteSynthesisUI
from gauss.synthesis.domains import SynthesisUI, SynthesisDomain
from gauss.synthesis.solver.engine import SynthesisEngine
from gauss.ui.grids.session import GridUISession
from gauss.ui.session import UISession

DOMAIN_DICT: Dict[str, Tuple[Type[SynthesisDomain], Type[SynthesisUI], Type[UISession]]] = {
    'pandas_lite': (PandasLiteSynthesisDomain, PandasLiteSynthesisUI, GridUISession)
}

ENGINE_CACHE: Dict[str, SynthesisEngine] = {}

SESSION_DICT: Dict[str, UISession] = {}


def _get_engine(domain_name: str) -> SynthesisEngine:
    if domain_name in ENGINE_CACHE:
        return ENGINE_CACHE[domain_name]

    try:
        ENGINE_CACHE[domain_name] = load_engine(domain_name)
        return ENGINE_CACHE[domain_name]

    except KeyError:
        raise ValueError(f"No engine known for domain {domain_name}.")


def get_session(session_id: str) -> UISession:
    return SESSION_DICT[session_id]


def start_synthesis(inputs: List[Any], domain: str = 'pandas_lite', engine: SynthesisEngine = None):
    if engine is None and domain is None:
        raise AssertionError(f"At least one of engine and domain has to be provided to start_synthesis.")

    if engine is None:
        try:
            domain_class, domain_ui_class, session_class = DOMAIN_DICT[domain]

        except KeyError:
            raise ValueError(f"Domain {domain} not supported")

        engine: SynthesisEngine = _get_engine(domain)

    output = widgets.Output()

    session: UISession = session_class(session_id=uuid.uuid4().hex,
                                       domain=domain_class(),
                                       domain_ui=domain_ui_class(),
                                       engine=engine,
                                       inputs=inputs,
                                       output_widget=output)

    SESSION_DICT[session.session_id] = session

    display(output)

    with output:
        session.display()
