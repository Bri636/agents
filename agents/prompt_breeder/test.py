""" Creating mutation prompts and such testing """
import streamlit as st
import pandas as pd
import numpy as np

from agents.prompt_breeder import create_population, init_run, run_for_n
from agents.prompt_breeder.mutation_prompts import mutation_prompts
from agents.prompt_breeder.thinking_styles import thinking_styles

from dotenv import load_dotenv
from rich import print
# import cohere

load_dotenv() # load environment variables


def dataframe_with_selections(mp_df, ts_df, mp_indices=None, ts_indices=None):
    # Add 'Select' column for marking selections
    mp_df_with_selections = mp_df.copy()
    ts_df_with_selections = ts_df.copy()
    mp_df_with_selections['Select'] = False
    ts_df_with_selections['Select'] = False

    # Use the provided indices or default to an empty list
    mp_selected_indices = mp_indices if mp_indices is not None else []
    ts_selected_indices = ts_indices if ts_indices is not None else []

    # Mark selections based on input indices
    mp_df_with_selections.loc[mp_selected_indices, 'Select'] = True
    ts_df_with_selections.loc[ts_selected_indices, 'Select'] = True

    # Filter and return only the selected rows
    mp_selected_rows = mp_df_with_selections[mp_df_with_selections['Select']]
    ts_selected_rows = ts_df_with_selections[ts_df_with_selections['Select']]

    return mp_selected_rows, ts_selected_rows

if __name__ == "__main__": 
    problem_description = """Solve the math word problem, giving your answer as an arabic numeral."""
    
    # Initialize dataframes
    ts_df = pd.DataFrame(thinking_styles)
    mp_df = pd.DataFrame(mutation_prompts)
    
    # Define selected indices (example values)
    mp_indices = [0, 2]  # Example indices for mutation prompts
    ts_indices = [1, 3]  # Example indices for thinking styles
    
    # Get selected rows based on provided indices
    mp_selected_rows, ts_selected_rows = dataframe_with_selections(mp_df, ts_df, mp_indices=mp_indices, ts_indices=ts_indices)

    # Convert selections to lists and create population
    population = create_population(
        tp_set=ts_selected_rows.iloc[:, 0].tolist(),  # Assuming 'thinking styles' are in the second column
        mutator_set=mp_selected_rows.iloc[:, 0].tolist(),  # Assuming 'mutation prompts' are in the second column
        problem_description=problem_description
    )

    print("Population created:", population)