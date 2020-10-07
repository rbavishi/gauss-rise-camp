"""
Useful routines to instantiate and use a synthesis engine.
"""

from gauss.synthesis.config import EngineConfig
from gauss.synthesis.deduction.engine import DeductionEngine
from gauss.synthesis.domains import SynthesisDomain
from gauss.synthesis.solver.engine import SynthesisEngine
from gauss.utilities.logutils import logger


@logger.catch(reraise=True)
def build_synthesis_engine(domain: SynthesisDomain, config: EngineConfig):
    logger.add(f"{config.path}/logs/run.log", mode="w")
    logger.opt(colors=True).debug(f"Building synthesis engine for domain <green>{domain.__class__.__name__}</green>.")
    logger.opt(colors=True).debug(f"Data and logs will be saved at <green>{config.path}</green>.")
    logger.opt(colors=True).debug(f"Using engine configuration: <green>{config}</green>")

    #  First build the deduction engine
    deduction_engine: DeductionEngine = DeductionEngine.build(domain, config)

    #  Now assemble the synthesis engine
    synthesis_engine = SynthesisEngine(domain=domain, config=config, deduction_engine=deduction_engine)
    return synthesis_engine
