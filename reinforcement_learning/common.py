from lartpc_game.agents.agents import BaseAgent, BaseMemoryAgent, ExperienceBuffer
from lartpc_game.game.game_ai import Lartpc2D

class RLAgent(BaseMemoryAgent, BaseAgent):
    def __init__(
            self,
            env: Lartpc2D,
            batch_size: int,
            trace_length: int,
            memory=None
    ):
        BaseMemoryAgent.__init__(self)
        BaseAgent.__init__(self, env)
        self.memory =  memory if memory is not None else ExperienceBuffer(buffer_size=4000)
        self.batch_size = batch_size
        self.trace_length = trace_length
