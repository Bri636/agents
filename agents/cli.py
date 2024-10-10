from __future__ import annotations

'''Cli interface for running autoprotocol'''

from pathlib import Path
import typer
from typing import Union, Optional, Tuple, Any
from typing_extensions import Annotated
from tqdm import tqdm
import os, sys, time
from textwrap import dedent
from argparse import ArgumentParser
import pprint as pp

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

@app.command(rich_help_panel="Workflow Generation")
def generate_experiment(
    workflow_name: Annotated[str, typer.Option(
        help='The name that you want to give your workflow',
        rich_help_panel="Input Experiment Files"
    )]
    
    
    
    
    ) -> None: 
    
    ...

def import_agent_packages() -> None: 
    
    from agents import generators
    from agents import prompts
    from agents import input_payloads
    
    from agents import llm_agents
    
    
if __name__=="__main__": 
    
    
    
    ...
    
    
    