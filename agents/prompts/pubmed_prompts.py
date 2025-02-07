""" Prompts for PubMedQA """


QUESTION={
  "instruction": """
You are an intelligent agent that is good at asking questions. 
Given a question, please decompose it into the next most logical sub-question. 
For each sub-question, an expert will give a sub-answer to that sub-question. 
When the original question is answerable, please start the subquestion with \"Question x.x: Now we can answer the question: \".
You are only allowed to ask questions. Do not give answers to your own questions.
  """,
  "interactive_examples": [
"""
Question {idx}: Are group 2 innate lymphoid cells ( ILC2s ) increased in chronic rhinosinusitis with nasal polyps or eosinophilia?
Question {idx}.1: How do elevated ILC2s in patients with CRSwNP contribute to the formation of nasal polyps in CRS?
Answer {idx}.1: As ILC2s are elevated in patients with CRSwNP, they may drive nasal polyp formation in CRS.
Question {idx}.2: In what ways do ILC2s influence eosinophil activation and survival, and how is this related to high tissue and blood eosinophilia in the Th2 immune response?
Answer {idx}.2: ILC2s are also linked with high tissue and blood eosinophilia and have a potential role in the activation and survival of eosinophils during the Th2 immune response.
Question {idx}.3: How does the presence of innate lymphoid cells in CRS enhance our understanding of the disease’s pathogenesis?
Answer {idx}.3: The association of innate lymphoid cells in CRS provides insights into its pathogenesis. The answer is #### Yes.
""",
"Question {idx}: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nQuestion {idx}.1: How much does Weng earn per minute?\nAnswer {idx}.1: Since Weng earns $12 an hour for babysitting, she earns $12 / 60 = $0.2 per minute. The answer is 0.2.\nQuestion {idx}.2: Now we can answer the question: How much did she earn?\nAnswer {idx}.2: Working 50 minutes, she earned $0.2 x 50 = $10. The answer is 10.",
"""
Question {idx}: Does vagus nerve contribute to the development of steatohepatitis and obesity in phosphatidylethanolamine N-methyltransferase deficient mice? 
Question {idx}.1: How do neuronal signals through the hepatic vagus nerve contribute to the development of steatohepatitis in HFD-fed Pemt(-/-) mice?
Answer {idx}.1: Neuronal signals via the hepatic vagus nerve contribute to the development of steatohepatitis and protection against obesity in HFD fed Pemt(-/-) mice. The answer is #### Yes.
"""
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

ANSWER={
  "instruction": """
You are an intelligent agent that is good at answering questions. 
An expert will ask you a sub-question at each round.
For each sub-question, please answer it in a complete sentence, ending with \"The answer is\". 
If it is your final answer, you must state the answer in the format \"#### Your final answer\". 
You are only allowed to give answers to questions.  
  """,
  "interactive_examples": [
"""
Question {idx}: Are group 2 innate lymphoid cells ( ILC2s ) increased in chronic rhinosinusitis with nasal polyps or eosinophilia?
Question {idx}.1: How do elevated ILC2s in patients with CRSwNP contribute to the formation of nasal polyps in CRS?
Answer {idx}.1: As ILC2s are elevated in patients with CRSwNP, they may drive nasal polyp formation in CRS.
Question {idx}.2: In what ways do ILC2s influence eosinophil activation and survival, and how is this related to high tissue and blood eosinophilia in the Th2 immune response?
Answer {idx}.2: ILC2s are also linked with high tissue and blood eosinophilia and have a potential role in the activation and survival of eosinophils during the Th2 immune response.
Question {idx}.3: How does the presence of innate lymphoid cells in CRS enhance our understanding of the disease’s pathogenesis?
Answer {idx}.3: The association of innate lymphoid cells in CRS provides insights into its pathogenesis. The answer is #### Yes.
""",
"Question {idx}: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nQuestion {idx}.1: How much does Weng earn per minute?\nAnswer {idx}.1: Since Weng earns $12 an hour for babysitting, she earns $12 / 60 = $0.2 per minute. The answer is 0.2.\nQuestion {idx}.2: Now we can answer the question: How much did she earn?\nAnswer {idx}.2: Working 50 minutes, she earned $0.2 x 50 = $10. The answer is 10.",
"""
Question {idx}: Does vagus nerve contribute to the development of steatohepatitis and obesity in phosphatidylethanolamine N-methyltransferase deficient mice? 
Question {idx}.1: How do neuronal signals through the hepatic vagus nerve contribute to the development of steatohepatitis in HFD-fed Pemt(-/-) mice?
Answer {idx}.1: Neuronal signals via the hepatic vagus nerve contribute to the development of steatohepatitis and protection against obesity in HFD fed Pemt(-/-) mice. The answer is #### Yes.
"""
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