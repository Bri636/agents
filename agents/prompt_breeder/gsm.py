import json
import re
import os
import random

"""
Format: 
[{'question': // text question //, 
    'answer': // Step-by-step answer, with "\ n" representing steps of thought, and ####space ... as the answer: 
}]

ex. 
[{'question': 
A cleaning company produces two sanitizer sprays. 
One spray kills 50% of germs, and another spray kills 25% of germs. 
However, 5% of the germs they kill are the same ones. 
What percentage of germs would be left after using both sanitizer sprays together?
    'answer': 
After the first spray kills 50% of germs, there will be 100 - 50 = <<100-50=50>>50% left.
The second spray kills 25%, but 5% have already been killed by the 50% spray, so it kills 25 - 5 = <<25-5=20>>20%.
After the second spray kills 20% of the remaining germs, there will be 50 - 20 = <<50-20=30>>30% left.
#### 30
}]

"""

def read_jsonl(path: str) -> list[dict[str, str]]:
    """
    Reads jsonl and returns it 
    """
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]
    
def get_examples(split):
    path = os.path.join("data/", f"{split}.jsonl")
    examples = read_jsonl(path)

    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"] + "<|endoftext|>")

    print(f"{len(examples)} {split} examples")
    return examples

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

def gsm_extract_answer(completion: str) -> str:
    """ 
    Parses through a string and returns the answer as a str
    
    Expects the answer in this format: 
    Answer is #### -567.89 ===> -567.89
    """
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS
    
def gsm_is_correct(model_completion: str, gt_example: dict[str, str]) -> bool:
    gt_answer = gsm_extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return gsm_extract_answer(model_completion) == gt_answer

# mine 
def batch_sample_qa_pairs(dataset: list[dict[str, str]], batch_size: int) -> list[dict[str, str]]: 
    """ Sample batches from gsm8k dataset """
    indices: list[int] = random.sample(range(len(dataset)), batch_size)
    sampled_data = [dataset[i] for i in indices]
    
    return sampled_data

def batch_gsm_extract_answer(completions: list[str]) -> list[str]: 
    """ Batch extracts answers """
    answers = list(map(gsm_extract_answer, completions))
    return answers

def batch_eval_gsm(parsed_model_answers: list[str], gsm_qa_pairs: list[dict[str, str]]) -> list[bool]: 
    """ Batch evals model answers to qa pairs from dataset"""
    return [gsm_is_correct(answer, example) for answer, example 
            in zip(parsed_model_answers, gsm_qa_pairs)]

if __name__=="__main__": 
    from typing import TypedDict, TypeVar, Tuple
    import io
    import pprint as pp
    from vllm import LLM, SamplingParams
    
    batch_size = 1
    dataset = read_jsonl('/lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/data/gsm.jsonl')
    samples: list[dict] = batch_sample_qa_pairs(dataset, batch_size)
    model_responses = [" The answer is #### 11 "] * batch_size
    completions = batch_gsm_extract_answer(model_responses)
    evals = batch_eval_gsm(completions, samples)
    
    class GSM8kPromptDict(TypedDict):
        """ Stores the Components for the Prompt """
        instruction: str
        interactive_examples: list[str]
        useful_examples: list[str]
        question_prefix: str
        subquestion_prefix: str
        overall_question_prefix: str
        answer_prefix: str
        
    Example = TypeVar('Example')
    
    def update_example_llama(prompt: GSM8kPromptDict, num_fsl_examples: int) -> list[dict[str, str]]:
        """ 
        Takes in a GSM8kPromptDict with a loaded instruction and fsl examples 
        and returns a list of {'user': ..., 'content': ...} messages for llama 
        """
        system_instruction = prompt['instruction']
        formatted_examples = []
        
        for idx, example_text in enumerate(prompt['interactive_examples']):
            formatted_example = [{"role": "system", "content": system_instruction}]
            # Extract the user question and answer pairs
            lines = example_text.strip().splitlines()
            # Add the main question to the formatted example
            main_question = next(line for line in lines if line.startswith("Question"))
            formatted_example.append({"role": "user", "content": main_question.strip().format(idx=idx + 1)})
            # Process sub-questions and answers
            for line in lines[1:]:
                if line.startswith("Question"):
                    formatted_example.append({"role": "user", "content": line.strip().format(idx=idx + 1)})
                elif line.startswith("Answer"):
                    formatted_example.append({"role": "assistant", "content": line.strip().format(idx=idx + 1)})
            formatted_examples.append(formatted_example)

        # Sample the specified number of formatted examples
        indices = random.sample(range(len(formatted_examples)), num_fsl_examples)
        selected_examples = [formatted_examples[i] for i in indices]

        return selected_examples

    
    def update_example(prompt: GSM8kPromptDict, num_fsl_examples: int) -> Tuple[str, list[str]]:
        system_instruction = prompt['instruction']
        formatted_examples = []
        for idx, example_text in enumerate(prompt['interactive_examples']):
            formatted_example = example_text.format(idx=idx + 1)
            formatted_examples.append(formatted_example + '\n')
            
        indices = random.sample(range(len(formatted_examples)), num_fsl_examples)
        formatted_examples = [formatted_examples[i] for i in indices]
        
        return system_instruction, formatted_examples
    
    prompt={
  "instruction": "Given a question, please decompose it into sub-questions. For each sub-question, please answer it in a complete sentence, ending with \"The answer is\". If it is your final answer, mark it with \"####\". When the original question is answerable, please start the subquestion with \"Question x.x: Now we can answer the question: \".",
  "interactive_examples": [
    "Question {idx}: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nQuestion {idx}.1: How many clips did Natalia sell in May?\nAnswer {idx}.1: Natalia sold 48 clips in April and half as many clips in May, so she sold 48 / 2 = 24 clips in May. The answer is 48.\nQuestion {idx}.2: Now we can answer the question: How many clips did Natalia sell altogether in April and May?\nAnswer {idx}.2: Natalia sold 48 clips in April and 24 clips in May, so altogether she sold 48 + 24 = 72 clips. The answer is #### 72.",
    "Question {idx}: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nQuestion {idx}.1: How much does Weng earn per minute?\nAnswer {idx}.1: Since Weng earns $12 an hour for babysitting, she earns $12 / 60 = $0.2 per minute. The answer is 0.2.\nQuestion {idx}.2: Now we can answer the question: How much did she earn?\nAnswer {idx}.2: Working 50 minutes, she earned $0.2 x 50 = $10. The answer is 10.",
    "Question {idx}: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\nQuestion {idx}.1: How much money does Betty have in the beginning?\nAnswer {idx}.1: In the beginning, Betty has only half of the money she needs, which is 100 / 2 = $50. The answer is 50.\nQuestion {idx}.2: How much money did Betty's grandparents give her?\nAnswer {idx}.2: Her grandparents gave her twice as much as her parents, so they gave her 15 * 2 = $30. The answer is 30.\nQuestion {idx}.3: Now we can answer the question: How much more money does Betty need to buy the wallet?\nAnswer {idx}.3: Now that she got $15 from her parents and $30 from her grandparents, she will need $100 - $15 - $30 = $55. Since she already has $50, she needs $55 - $50 = $5 more. The answer is #### 5.",
    "Question {idx}: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?\nQuestion {idx}.1: How many pages did Julie read today?\nAnswer {idx}.1: Julie read twice as many pages as yesterday, so she read 12 * 2 = 24 pages The answer is 24.\nQuestion {idx}.2: How many pages did Julie read since yesterday?\nAnswer {idx}.2: Since yesterday, Julie read 12 + 24 = 36 pages. The answer is 36.\nQuestion {idx}.3: How many pages are left to be read?\nAnswer {idx}.3: There are 120 - 36 = 84 pages left to be read. The answer is 84.\nQuestion {idx}.4: Now we can answer the question: How many pages should she read?\nAnswer {idx}.4: She wants to read half of the remaining pages, so she should read 84 / 2 = 42 pages. The answer is #### 42.",
    "Question {idx}: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?\nQuestion {idx}.1: How many pages does he write every week?\nAnswer {idx}.1: James writes a 3-page letter to 2 different friends twice a week, so he writes 3 * 2 * 2 = 12 pages every week. The answer is 12.\nQuestion {idx}.2: How many weeks are there in a year?\nAnswer {idx}.2: There are 52 weeks in a year. The answer is 52.\nQuestion {idx}.3: Now we can answer the question: How many pages does he write a year?\nAnswer {idx}.3: James writes 12 pages every week, so he writes 12 * 52 = 624 pages a year. The answer is 624.",
    "Question {idx}: Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden?\nQuestion {idx}.1: How many purple flowers are there?\nAnswer {idx}.1: There are 80% more purple flowers than yellow flowers, so there are 10 * 1.8 = 18 purple flowers. The answer is 18.\nQuestion {idx}.2: How many yellow and purple flowers are there in total?\nAnswer {idx}.2: There are 10 yellow flowers and 18 purple flowers, so there are 10 + 18 = 28 yellow and purple flowers. The answer is 28.\nQuestion {idx}.3: How many green flowers are there?\nAnswer {idx}.3: There are 25% as many green flowers as there are yellow and purple flowers, so there are 28 * 0.25 = 7 green flowers. The answer is 7.\nQuestion {idx}.4: Now we can answer the question: How many flowers does Mark have in his garden?\nAnswer {idx}.4: Mark has 10 yellow flowers, 18 purple flowers, and 7 green flowers, so he has 10 + 18 + 7 = 35 flowers in his garden. The answer is #### 35.",
    "Question {idx}: Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?\nQuestion {idx}.1: How many slices do the large pizzas have?\nAnswer {idx}.1: He buys 2 large pizzas, so he has 2 * 16 = 32 slices. The answer is 32.\nQuestion {idx}.2: How many slices do the small pizzas have?\nAnswer {idx}.2: He buys 2 small pizzas, so he has 2 * 8 = 16 slices. The answer is 16.\nQuestion {idx}.3: How many pieces does he eat that day?\nAnswer {idx}.3: Now we can answer the question: There are 32 slices from the large pizzas and 16 slices from the small pizzas, so he eats 32 + 16 = 48 pieces that day. The answer is #### 48.",
    "Question {idx}: Ken created a care package to send to his brother, who was away at boarding school. Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds. Then, he added enough brownies to cause the weight to triple. Next, he added another 2 pounds of jelly beans. And finally, he added enough gummy worms to double the weight once again. What was the final weight of the box of goodies, in pounds?\nQuestion {idx}.1: What was the weight of the box after Ken poured jelly beans for the first time?\nAnswer {idx}.1: Ken poured jelly beans into the box until the weight was 2 pounds, so the weight of the box was 2 pounds. The answer is 2.\nQuestion {idx}.2: What was the weight of the box after Ken added brownies?\nAnswer {idx}.2: Ken aadded enough brownies to cause the weight to triple, so the weight of the box was 2 * 3 = 6 pounds. The answer is 6.\nQuestion {idx}.3: What was the weight of the box after Ken added jelly beans for the second time?\nAnswer {idx}.3: He added another 2 pounds of jelly beans, which means the weight of the box was 6 + 2 = 8 pounds. The answer is 8.\nQuestion {idx}.4: Now we can answer the question: What was the final weight of the box of goodies, in pounds?\nAnswer {idx}.4: Finally, he added enough gummy worms to double the weight once again, so the weight of the box was 8 * 2 = 16 pounds. The answer is #### 16.",
    "Question {idx}: Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of $200 and spent $30 on a button-up shirt, $46 on suit pants, $38 on a suit coat, $11 on socks, and $18 on a belt. She also purchased a pair of shoes, but lost the receipt for them. She has $16 left from her budget. How much did Alexis pay for the shoes?\nQuestion {idx}.1: How much did Alexis pay for everything else?\nAnswer {idx}.1: Alexis spent $30 on a button-up shirt, $46 on suit pants, $38 on a suit coat, $11 on socks, and $18 on a belt, so she spent 30 + 46 + 38 + 11 + 18 = $143 on everything else. The answer is 143.\nQuestion {idx}.2: How much money did Alexis spend in total?\nAnswer {idx}.2: Alexis had a budget of $200 and finally there was $16 left, so she spent 200 - 16 = $184 in total. The answer is 184.\nQuestion {idx}.3: Now we can answer the question: How much did Alexis pay for the shoes?\nAnswer {idx}.3: Alexis spent $143 on everything else, so she spent 184 - 143 = $41 on the shoes. The answer is #### 41.",
    "Question {idx}: Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works 10 hours every day for 5 days, how much money does she make?\nQuestion {idx}.1: How much does Tina make in an 8-hour shift?\nAnswer {idx}.1: Tina makes $18.00 an hour, so she makes 18 * 8 = $144.00 in an 8-hour shift. The answer is 144.\nQuestion {idx}.2: How many hours of overtime does Tina get?\nAnswer {idx}.2: Tina works 10 hours every day for 5 days, so she works 10 * 5 = 50 hours. Since she works 8 hours every day, she gets 50 - 8 * 5 = 10 hours of overtime. The answer is 10.\nQuestion {idx}.3: How much is her hourly overtime wage?\nAnswer {idx}.3: Her hourly overtime wage is 18 + 18 / 2 = $27.00. The answer is 27.\nQuestion {idx}.4: How much does Tina make in overtime each day?\nAnswer {idx}.4: Tina works 10 hours a day, and 8 hours of that is paid at her regular hourly wage, so she makes 10 - 8 = 2 hours of overtime every day. Since her hourly overtime wage is $27.00, she makes 27 * 2 = $54.00 in overtime each day. The answer is 54.\nQuestion {idx}.5: How much does Tina make each day?\nAnswer {idx}.5: Tina makes $144.00 in an 8-hour shift and $54.00 in overtime, so she makes 144 + 54 = $198.00 each day. The answer is 198.\nQuestion {idx}.6: Now we can answer the question: How much money does she make?\nAnswer {idx}.6: Tina works 5 days a week, so she makes 198 * 5 = $990.00. The answer is #### 990."
  ],
  "useful_examples": [
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    ""
  ],
  "question_prefix": "Question {idx}: {question}",
  "subquestion_prefix": "Question {idx}.{sub_idx}:",
  "overall_question_prefix": "Now we can answer the question:",
  "answer_prefix": "Answer {idx}.{sub_idx}:"
}
    
    # instruction, fsl_prompts = update_example(prompt, 3)
    fsl_prompts = update_example_llama(prompt, 3)
    
    conversations:list[list[dict]] = fsl_prompts
    conversations[0].append({'role': 'user', 'content': f'Question 2: {samples[0]["question"]}'})
    # conversations[0].append({'role': 'assistant', 'content': 'Question 2.1: What is the prime factorization of 56?'})
    # conversations[0].append({'role': 'user', 'content': 'Answer 2.1: The prime factorization of 56 is 2 * 2 * 2 * 7 = 2^3 * 7 The answer is 2^3 * 7.'})
    test_input = conversations[0]
    # llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
    breakpoint()
    llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
    sampling_params = SamplingParams(temperature=0.5, max_tokens=5000)
    
    Q = re.compile(r"Question (\-?[0-9\.\,]+)")
    A = re.compile(r"Answer (\-?[0-9\.\,]+)")
    FA = re.compile(r"#### (\-?[0-9\.\,]+)")
    other = []
    for i in range(20): 
        
        outputs = llm.chat(test_input,
                    sampling_params=sampling_params,
                    use_tqdm=False)
        output = outputs[0].outputs[0].text
        
        match_Q = Q.search(output)
        match_A = A.search(output)
        match_FA = FA.search(output)
        
        if match_Q: 
            match_Q_str = match_Q.group(1).strip()
            match_Q_str = match_Q_str.replace(",", "")
            test_input.append({'role': 'assistant', 'content': output})
        
        elif match_A: 
            match_A_str = match_A.group(1).strip()
            match_A_str = match_A_str.replace(",", "")    
            test_input.append({'role': 'user', 'content': output})
            
        elif match_FA: 
            match_FA_str = match_FA.group(1).strip()
            match_FA_str = match_FA_str.replace(",", "")   
            final = match_FA_str
            break
        else: 
            other.append(output)
            
    breakpoint()
    if final: 
        
        breakpoint()
    
    breakpoint()