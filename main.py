import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def parse_qa_from_file(file_path):
    """
    Parses questions and answers from a given text file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    qa_pairs = content.split('\n\n')
    parsed_data = []

    for pair in qa_pairs:
        if pair.strip():
            question, answer = pair.split('\n')
            parsed_data.append((question.split(': ')[1], answer.split(': ')[1]))

    return parsed_data

def generate_chain_of_thought(question, answer):
    """
    Generates a chain of thought for a given question and answer.
    """
    prompt = (f"Act as a very intelligent rationale sequence generator. Your task is to create Sequences Of Rationales "
              f"(SOR-s): 1 correct and 3 incorrect (with mistakes in the correct SOR). It is inspired by mathematical "
              f"fallacy, where people create convincing yet incorrect proofs.\nUse this example format for incorrect. "
              f"Incorrect rationale may be any of them.\nDo not mention that some rationales are incorrect inside SOR text.\n\n"
              f"SOR 1 (incorrect)\nRationale 1:\nRationale 2 (incorrect):\n...\nRationale N:\nConclusion:\n\n"
              f"SOR 2 (correct)\n...\n\nNow proceed with this task:\n{question}\nIts answer is:\n{answer}\n")
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].text

def analyze_chain_of_thought(question, chain_of_thought):
    """
    Analyzes the chain of thought for a given question.
    """
    prompt = (f"Act as a very intelligent rationale discriminator. Your task is to figure out if there are any "
              f"incorrect statements in a Chain of Thought of a generator. Avoid false positives. There is one correct "
              f"Chain of Thought.\nHere is a list of logic rules for you to remember:\n\n"
              f"Modus Ponens: If P then Q; P; therefore, Q.\n"
              f"Modus Tollens: If P then Q; not Q; therefore, not P.\n"
              f"Hypothetical Syllogism: If P then Q; if Q then R; therefore, if P then R.\n"
              f"Disjunctive Syllogism: P or Q; not P; therefore, Q.\n"
              f"Conjunction: P; Q; therefore, P and Q.\n"
              f"Simplification: P and Q; therefore, P.\n"
              f"Addition: P; therefore, P or Q.\n"
              f"Biconditional Introduction: P if and only if Q; therefore, if P then Q, and if Q then P.\n"
              f"Biconditional Elimination: P if and only if Q; P; therefore, Q.\n"
              f"Constructive Dilemma: P or Q; if P then R; if Q then S; therefore, R or S.\n"
              f"Destructive Dilemma: P or Q; if P then not R; if Q then not S; therefore, not R or not S.\n"
              f"Chain Rule: If P then Q; if Q then R; P; therefore, R.\n"
              f"Law of Excluded Middle: Either P or not P.\n"
              f"Law of Non-Contradiction: Not both P and not P.\n"
              f"Double Negation: Not not P; therefore, P.\n"
              f"Transposition: If P then Q; therefore, if not Q then not P.\n"
              f"Material Implication: If P then Q; therefore, not P or Q.\n"
              f"Exportation: If P and Q then R; therefore, if P then (if Q then R).\n"
              f"Importation: If P then (if Q then R); therefore, if P and Q then R.\n"
              f"De Morgan's Theorem: Not (P and Q); therefore, not P or not Q (and vice versa).\n\n"
              f"Now proceed with this task:\n{question}\nChains of Thought:\n{chain_of_thought}\n")
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].text

model = 'gpt-3.5-turbo-0613'
input_file_path = 'dataset.txt'
generator_output_path = 'generator_output.txt'
discriminator_output_path = 'discriminator_output.txt'

qa_pairs = parse_qa_from_file(input_file_path)
with open(generator_output_path, 'w', encoding='utf-8') as gen_file, open(discriminator_output_path, 'w', encoding='utf-8') as disc_file:
    for idx, (question, answer) in enumerate(qa_pairs):
        chain_of_thought = generate_chain_of_thought(question, answer)
        gen_file.write(f"Index: {idx}\nQuestion: {question}\nAnswer: {answer}\nChain of Thought: {chain_of_thought}\n\n")

        analysis = analyze_chain_of_thought(question, chain_of_thought)
        disc_file.write(f"Index: {idx}\nQuestion: {question}\nChains of Thought: {chain_of_thought}\nAnalysis: {analysis}\n\n")