
from agents.utils import register_strategy

class RewardRolloutStrategies:
    """ Strategies for assigning rewards for terminal and non-terminal phases """ 
    
    strategies = {}
    
    @register_strategy(strategies, name='base')
    def base(win: bool,
             win_reward: float = 100,
             lose_reward: float = -50
             ) -> float:
        """ Classic Reward Shaping for Win-Loss """
        return win_reward if win else lose_reward
    
    
       # @classmethod
    # def execute_strategy(cls, win: bool, strategy: str) -> tuple[list["BTMCTSNode"], float]:
    #     """ Interface for executing strategies."""
    #     strategy_func: Callable = cls.strategies.get(strategy)
    #     if not strategy_func:
    #         raise ValueError(f"Strategy '{strategy}' does not exist. Choose from {list(cls.strategies.keys())}")
    #     instance = cls()
    #     return strategy_func(instance, root, cum_reward_func) 
    

# class RewardStrategy: 
#     """ Strategies for generating """
#     strategies = {}
    
#     def log_probs(self, answer_prompt, generator, **kwargs): 
#         """ Assigns Rewards based on log probs"""
        
#     def confidence(self, answer_prompt, generator: BaseLLMGenerator, num_samples: int, **kwargs): 
#         """ Perform k samples and then find proportion of most frequent answer """
        
#         k_shot_prompts = [answer_prompt] * num_samples
#         actions = [generator.generate(prompt) for prompt in k_shot_prompts]
        
#         return actions
    
#     def mc_rollout(self): 
#         """ Do rollout on that """