""" Basic Chat Prompt for Chat-Based Modeling """

from __future__ import annotations

from collections import UserList
from typing import Optional, Literal, Self, Any, Union, TypedDict, List
from dataclasses import dataclass, asdict
import copy
import pprint as pp

@dataclass
class ChatMessage:
    """ Dataclass interface for OpenAI-like chat prompts {'role': str, 'content': str} """
    role: Literal['system', 'user', 'assistant']
    content: str
    
    def __post_init__(self):
        if self.role not in {'system', 'user', 'assistant'}:
            raise ValueError(f"Invalid role: {self.role}. Must be one of ['system', 'user', 'assistant'].")
    
# TODO: to construct ChatMessageSequnece, you give it a dict, a list of dict, or another ChatMessage or ChatMessageSequence
# this is model outputs and user constructed data is probably going to be a list of dicts
# it does not make sense to coerce into a ChatMessage or ChatMessageSequence, then validate it as a ChatMessage
# that is circular logic 
    
# IDEA: SAME AS USER LIST, JUST A list[ChatMessage]
class ChatMessageSequence(UserList[ChatMessage]): 
    """ Sequence of ChatMessage dicts that represents sequential chat history list[{'role': str, 'content': str}] """
    def __init__(self, 
                 messages: list[dict],
                 few_shot_learning_sequence: Optional[list[dict]] = None,
                 allow_overwrite: bool = False, # allow in-place ops
                 verbose: bool = False,
                 **kwargs
                 ) -> None:
        
        # set attributes
        self.allow_overwrite = allow_overwrite
        self.verbose = verbose
            
        base_messages = [self._validate_chat_message(message) 
                         for message in messages]
        
        if few_shot_learning_sequence: 
            base_messages += [self._validate_chat_message(message) 
                              for message in few_shot_learning_sequence]
            
        super().__init__(message for message in base_messages)
        
    def _validate_chat_message(self, message: dict[str, str]) -> ChatMessage:
        """ Ensure an atomic message is a valid ChatMessage object."""
        if isinstance(message, dict):
            if self.verbose:
                print("Validating as ChatMessage object...")
            return ChatMessage(**message)
        else:
            raise ValueError(f"Invalid message type: {type(message)}. Must be ChatMessage")
        
    # @classmethod
    # def build(cls, messages: dict[str, str] | list[dict[str, str]], **kwargs) -> ChatMessageSequence: 
    #     """ Builds Sequence from another ChatMessage or ChatMessageSequence """
    #     msg_sequence = cls(messages, **kwargs)
    #     return msg_sequence
        
    # def history(self) -> ChatMessageSequence: 
    #     return copy.deepcopy(self.data)
    
    # def __setitem__(self, index: int, message: ChatMessageLike) -> None: 
    #     self.data[index] = self._validate_chat_message(message)
        
    # def __getitem__(self, index: int) -> ChatMessage: 
    #     """ Deepcopy so we dont accidentally overwrite the chat message"""
    #     if self.allow_overwrite: 
    #         return self.data[index]
    #     return copy.deepcopy(self.data[index])
    
    # def append(self, message: ChatMessageLike):
    #     return super().append(self._validate_chat_message(message))
    
    # def extend(self, messages: list[ChatMessageLike] | ChatMessageSequence):
    #     messages = [self._validate_chat_message(message) for message in messages]
    #     return super().extend(messages)
    
    # def clear(self): 
    #     self.data = []
    # # TODO: list[ChatMessageLike] type union with ChatMessageSequence
    # def insert(self, idx: int, messages: ChatMessageLike | list[ChatMessageLike] | ChatMessageSequence):
    #     if isinstance(messages, ChatMessageSequence) or isinstance(messages, list): 
    #         if self.verbose: 
    #             print(f'Unfolding sequence of messages')
    #         for message in messages: 
    #             super().insert(idx, self._validate_chat_message(message))
    #             idx += 1
    #     else: 
    #         super().insert(idx, self._validate_chat_message(messages))
            
    def as_list(self) -> List[dict[str, Any]]:
        """Convert the sequence to a list of dictionaries (JSON-serializable)."""
        return [asdict(msg) for msg in self]
    
    def __repr__(self) -> str:
        return f"{pp.pformat(self.data)}"

    def __str__(self) -> str:
        return pp.pformat(self.data)
    
    
# ChatMessageLikeSequence = Union[ChatMessageSequence, list[ChatMessageLike]]
        
BatchChatMessageSequence = list[ChatMessageSequence]
""" Batch Sequence of ChatMessageSequences that represents list[list[{'role': str, 'content': str}]]"""

def system_map(system_prompt: str, user_prompt: str) -> ChatMessageSequence:
    return ChatMessageSequence([
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ])
    
    
def o1_map(system_prompt: str, user_prompt: str) -> ChatMessageSequence:
    return ChatMessageSequence([
        {'role': 'user', 'content': system_prompt + '\n' + user_prompt}
    ])