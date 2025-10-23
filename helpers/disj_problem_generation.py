import random
import numpy as np
import re


prop_var_range = 26

# generate rules, facts, CoT and answer

special_counter = 1


CONJ_token = prop_var_range + special_counter
special_counter+=1
DISJ_token = prop_var_range + special_counter
special_counter+=1
NEGAT_token = prop_var_range + special_counter
special_counter+=1

IF_token = prop_var_range + special_counter
special_counter+=1
THEN_token = prop_var_range + special_counter
special_counter+=1

START_RULE_token = prop_var_range + special_counter
special_counter+=1
END_RULE_token = prop_var_range + special_counter
special_counter+=1

START_FACT_token = prop_var_range + special_counter
special_counter+=1
END_FACT_token = prop_var_range + special_counter
special_counter+=1


TRUE_token = prop_var_range + special_counter
special_counter+=1
FALSE_token = prop_var_range + special_counter
special_counter+=1
UNDETERMINED_token = prop_var_range + special_counter
special_counter+=1

START_QUERY_token = prop_var_range + special_counter
special_counter+=1
END_QUERY_token = prop_var_range + special_counter
special_counter+=1

SEP_token = prop_var_range + special_counter
special_counter+=1
ANSWER_token = prop_var_range + special_counter
special_counter+=1

SEMICOLON_token = prop_var_range + special_counter
special_counter+=1
EOS_token = prop_var_range + special_counter
special_counter+=1

special_token_dict = {"CONJ": CONJ_token,
                      "DISJ": DISJ_token,
                      "NEGAT": NEGAT_token,
                      "IF": IF_token,
                      "THEN": THEN_token,
                      "START_RULE": START_RULE_token,
                      "END_RULE": END_RULE_token,
                      "START_FACT": START_FACT_token,
                      "END_FACT": END_FACT_token,
                      "TRUE": TRUE_token,
                      "FALSE": FALSE_token,
                      "UNDETERMINED": UNDETERMINED_token,
                      "START_QUERY": START_QUERY_token,
                      "END_QUERY": END_QUERY_token,
                      "SEP": SEP_token,
                      "ANSWER": ANSWER_token,
                      ";": SEMICOLON_token,
                      "EOS": EOS_token}

# Generate the basic information needed for a disjunction + linear chain problem.
def sample_disjunction_chain(depth, shuffle=True, linear_only=False, vars=None, truths=None, LO_first=True):
  if vars is None:
    vars = np.random.choice(list(range(prop_var_range)), depth*3, replace=False)

  truth_values = np.random.choice(["TRUE", "FALSE"], 2)
  truth_value_distraction = np.random.choice(["TRUE", "FALSE"], 2)
  if truths is not None:
    truth_values[0] = truths[0]
    truth_values[1] = truths[1]
    truth_value_distraction[0] = truths[2]
  merge_depth = depth - 2
  vars_1 = vars[:depth]
  vars_2 = vars[depth:2*depth]
  vars_3 = vars[2*depth:]
  random.shuffle(vars)
  rules1 = []
  rules3 = []
  facts = [[vars_1[0],truth_values[0]] , [vars_2[0], truth_values[1]],
   [vars_3[0], truth_value_distraction[0]]]
  cot_complete = []
  cot_complete_linear = []
  global_truth_val = None

  # to return for more modular understanding of the logic problem
  queried_rule = None
  correct_fact_LO = None
  correct_fact_lin = None

  if linear_only:
    solve_disjunction = 0
    query = vars_3[-1]
  else:
    solve_disjunction = 1
    query = vars_1[-1]

    
    

  if solve_disjunction == 1:
    if truth_values[0] == "TRUE" and truth_values[1] == "TRUE":
      cot_complete.append([vars_1[0], special_token_dict[truth_values[0]]])
      correct_fact_LO = [vars_1[0], special_token_dict[truth_values[0]]]
    elif truth_values[0] == "TRUE" and truth_values[1] == "FALSE":
      cot_complete.append([vars_1[0], special_token_dict[truth_values[0]]])
      correct_fact_LO = [vars_1[0], special_token_dict[truth_values[0]]]
    elif truth_values[0] == "FALSE" and truth_values[1] == "TRUE":
      cot_complete.append([vars_2[0], special_token_dict[truth_values[1]]])
      correct_fact_LO = [vars_2[0], special_token_dict[truth_values[1]]]
    elif truth_values[0] == "FALSE" and truth_values[1] == "FALSE":
      cot_complete.append([vars_1[0], special_token_dict[truth_values[0]],
                          vars_2[0], special_token_dict[truth_values[1]]])
      correct_fact_LO = [vars_1[0], special_token_dict[truth_values[0]],
                          vars_2[0], special_token_dict[truth_values[1]]]

  for de in range(len(vars_1)-1):
    if de < merge_depth:
      rules1.append([vars_1[de], special_token_dict['THEN'], vars_1[de+1]])
      rules1.append([vars_2[de], special_token_dict['THEN'], vars_2[de+1]])

      if solve_disjunction == 1:
        # (False, True)=> second chain should be in CoT
        if truth_values[0] == "FALSE" and truth_values[1] == "TRUE":
          ans = [vars_2[de], special_token_dict['THEN'], vars_2[de+1],
                special_token_dict[';'], vars_2[de+1], special_token_dict['TRUE']]
          cot_complete.append(ans)
        # (True, False)=> first chain should be in CoT
        elif truth_values[0] == "TRUE" and truth_values[1] == "FALSE":
          ans = [vars_1[de], special_token_dict['THEN'], vars_1[de+1],
                special_token_dict[';'], vars_1[de+1], special_token_dict['TRUE']]
          cot_complete.append(ans)
        # (True, True)=> pick chain 1
        elif truth_values[0] == "TRUE" and truth_values[1] == "TRUE":
          ans1 = [vars_1[de], special_token_dict['THEN'], vars_1[de+1],
                  special_token_dict[';'], vars_1[de+1], special_token_dict['TRUE']]
          cot_complete.append(ans1)
        # (False, False)=> evaluate both chains, so they both should be in CoT
        else:
          ans1 = [vars_1[de], special_token_dict['THEN'], vars_1[de+1],
                  special_token_dict[';'], vars_1[de+1], special_token_dict['UNDETERMINED']]
          ans2 = [vars_2[de], special_token_dict['THEN'], vars_2[de+1],
                  special_token_dict[';'], vars_2[de+1], special_token_dict['UNDETERMINED']]
          cot_complete.append(ans1)
          cot_complete.append(ans2)
    # Now we are at the merging point of disjunction
    elif de == merge_depth:
      if truth_values[0] == "TRUE" or truth_values[1] == "TRUE":
        global_truth_val = "TRUE"
      else:
        global_truth_val = "UNDETERMINED"
      # allow either A OR B or B OR A in the solution and rule
      ans1 = [vars_1[de], special_token_dict["DISJ"], vars_2[de], special_token_dict['THEN'], vars_1[de+1],
              special_token_dict[';'], vars_1[de+1], special_token_dict[global_truth_val]]
      ans2 = [vars_2[de], special_token_dict["DISJ"], vars_1[de], special_token_dict['THEN'], vars_1[de+1],
              special_token_dict[';'], vars_1[de+1], special_token_dict[global_truth_val]]
      rule1_curr = [vars_1[de], special_token_dict["DISJ"], vars_2[de], special_token_dict['THEN'], vars_1[de+1]]
      rule2_curr = [vars_2[de], special_token_dict["DISJ"], vars_1[de], special_token_dict['THEN'], vars_1[de+1]]
      coin_flip1 = np.random.choice([1, 0])  # for CoT and rules
      if coin_flip1 == 1:
        if solve_disjunction == 1:
          cot_complete.append(ans1)
        rules1.append(rule1_curr)
      else:
        if solve_disjunction == 1:
          cot_complete.append(ans2)
        rules1.append(rule2_curr)
    # past the merging point
    else:
      if solve_disjunction == 1:
        cot_complete.append([vars_1[de], special_token_dict["THEN"], vars_1[de+1],
                           special_token_dict[';'], vars_1[de+1], special_token_dict[global_truth_val]])
      rules1.append([vars_1[de], special_token_dict['THEN'], vars_1[de+1]])


  cot_complete_linear.append([vars_3[0], special_token_dict[truth_value_distraction[0]]])
  correct_fact_lin = [vars_3[0], special_token_dict[truth_value_distraction[0]]]
  for de in range(len(vars_3)-1):
    rules3.append([vars_3[de], special_token_dict['THEN'], vars_3[de+1]])
    if truth_value_distraction[0] == "TRUE":
      cot_complete_linear.append([vars_3[de], special_token_dict['THEN'], vars_3[de+1],
                          special_token_dict[';'], vars_3[de+1], special_token_dict["TRUE"]])
    else:
      cot_complete_linear.append([vars_3[de], special_token_dict['THEN'], vars_3[de+1],
                          special_token_dict[';'], vars_3[de+1], special_token_dict["UNDETERMINED"]])

  if LO_first:
    rules = rules1 + rules3
  else:
    rules = rules3 + rules1
  if shuffle:
    random.shuffle(facts)


  result = [{'rules': rules,
            'facts': facts,
            'query': vars_1[-1],
            'cot': cot_complete,
            'answer': global_truth_val,
            'prop_vars': vars,
            'queried_rule': rules1[0],
            'correct_fact': correct_fact_LO},
            {'rules': rules,
            'facts': facts,
            'query': vars_3[-1],
            'cot': cot_complete_linear,
            'answer': global_truth_val,
            'prop_vars':vars,
            'queried_rule': rules3[0],
            'correct_fact': correct_fact_lin}]

  return result


def generate_one_sample(sample_rules, sample_facts, sample_query, sample_cot):
  sample_context = []
  # state the RULES
  for curr_rule in sample_rules:
    sample_context += curr_rule
    sample_context.append(special_token_dict['SEP'])

  # now state the FACTS
  sample_context.append(special_token_dict['START_FACT'])
  for facts in sample_facts:
    sample_context.append(facts[0])
    sample_context.append(special_token_dict[facts[1]])
    sample_context.append(special_token_dict['SEP'])
  sample_context.append(special_token_dict['END_FACT'])

  # state the QUERY variable
  sample_context.append(special_token_dict['START_QUERY'])
  sample_context.append(sample_query)
  sample_context.append(special_token_dict['SEP'])
  sample_context.append(special_token_dict['END_QUERY'])

  sample_context.append(special_token_dict['ANSWER'])

  sample_context_len = len(sample_context)

  # now construct the CoT answer
  sample_answer = []
  for i, curr_rule in enumerate(sample_cot):
    sample_answer += curr_rule
    sample_answer.append(special_token_dict['SEP'])

  return sample_context, sample_answer, sample_context_len



# Helper for converting integer tokens to simple english

integer_to_english_letters = {0:"A", 1:"B", 2:"C", 3:"D", 4:"E", 5:"F",
                              6:"G", 7:"H", 8:"I", 9:"J", 10:"K", 11:"L",
                              12:"M", 13:"N", 14:"O", 15:"P", 16:"Q", 17:"R",
                              18:"S", 19:"T", 20:"U", 21:"V", 22:"W",
                              23: "X", 24:"Y", 25:"Z",
                              THEN_token:"implies", START_FACT_token:"Facts:",
                              START_QUERY_token:"Question:", END_FACT_token:"FA_e",
                              END_QUERY_token:"Q_e", ANSWER_token:"ANS",
                              TRUE_token:"is true", FALSE_token:"is false",
                              SEMICOLON_token: ";", DISJ_token: "or", CONJ_token: "and",
                              SEP_token: ".", UNDETERMINED_token: 'is undetermined'}

def convert_to_english(int_str, is_question=True):
  if is_question:
    sample_eng_curr = ["Rules:"]
  else:
    sample_eng_curr = []
  for n in range(len(int_str)):
    if int_str[n] < prop_var_range:
      sample_eng_curr.append(integer_to_english_letters[int_str[n]])
    elif int_str[n] not in integer_to_english_letters:
      sample_eng_curr.append("UND")
    elif int_str[n] == END_FACT_token \
    or int_str[n] == END_QUERY_token \
    or int_str[n] == ANSWER_token:
      continue
    elif int_str[n] == START_QUERY_token:
      sample_eng_curr.append("Question: state the truth value of")
    else:
      sample_eng_curr.append(integer_to_english_letters[int_str[n]])
      '''if is_question == False and (int_str[n] == TRUE_token or int_str[n] == FALSE_token) \
       and int_str[n+1] < prop_var_range:
        sample_eng_curr.append("and")'''
  sample_eng_str_curr = " ".join(sample_eng_curr)
  return_str = re.sub(r'\s(?=[\.,:;])', "", sample_eng_str_curr)
  return sample_eng_curr, return_str



def sample_context_and_answer_pairs_EXAMPLES(num_samples, length_of_chain = 6, print_answer = True, randomness=True, LO=None):
  output_string = ""
  gt_string = ""
  shuffle_rules_and_facts = True
  LO_position = np.random.choice(range(num_samples), size=int(num_samples/2), replace=False)
  if set(LO_position) == set(range(int(num_samples/2))):
    LO_position = np.random.choice(range(num_samples), size=int(num_samples/2), replace=False)
  LO_first_indices = np.random.choice(range(num_samples), size=int(num_samples/2), replace=False)
  LO_first = []
  for i in range(num_samples):
    if i in LO_first_indices:
      LO_first.append(True)
    else:
      LO_first.append(False)

  LO_chain_truth_idx = np.random.choice([0, 1])
  LO_base_truths = [["TRUE", "FALSE"], ["TRUE", "TRUE"]]
  for i in range(num_samples):
    if i in LO_position:
      lin_chain_truth = np.random.choice(["TRUE", "FALSE"])
      LO_chain_truth = LO_base_truths[LO_chain_truth_idx]

      sample_dict_LO, _ = sample_disjunction_chain(length_of_chain, shuffle=shuffle_rules_and_facts, truths=["TRUE", "FALSE", lin_chain_truth], linear_only=False, LO_first=LO_first[i])
      sample_dict = sample_dict_LO
    else:
      lin_chain_truth = "TRUE"
      _, sample_dict_lin = sample_disjunction_chain(length_of_chain, shuffle=shuffle_rules_and_facts, truths=["TRUE", "FALSE", lin_chain_truth], linear_only=False, LO_first=LO_first[i])
      sample_dict = sample_dict_lin
    sample_rules = sample_dict['rules']
    sample_facts = sample_dict['facts']
    sample_query = sample_dict['query']
    sample_cot = sample_dict['cot']
    sample_prop_vars = sample_dict['prop_vars']

    sample_context, sample_answer, _ = generate_one_sample(sample_rules, sample_facts, sample_query, sample_cot)

    output_string += "" + convert_to_english(sample_context)[1] + " "
    gt_string = convert_to_english(sample_answer, is_question=False)[1]
    if print_answer:
      output_string += "Answer: " + gt_string + "\n"
    else:
      output_string += ""

  return output_string, gt_string



def sample_context_and_answer_pairs_QUESTION(num_samples, length_of_chain = 2, print_answer = True, randomness=True):
  output_string_LO = ""
  output_string_lin = ""
  gt_string_LO = ""
  gt_string_lin = ""
  shuffle_rules_and_facts = True
  for i in range(num_samples):
    LO_first = np.random.choice([True, False])
    if randomness:
      sample_dict_LO, sample_dict_lin = sample_disjunction_chain(length_of_chain, shuffle=shuffle_rules_and_facts, LO_first=LO_first)
    else:
      lin_chain_truth = "TRUE"
      sample_dict_LO, sample_dict_lin = sample_disjunction_chain(length_of_chain, shuffle=shuffle_rules_and_facts, truths=["TRUE", "FALSE", lin_chain_truth], LO_first=LO_first)

    # first generate the LO chain
    sample_rules = sample_dict_LO['rules']
    sample_facts = sample_dict_LO['facts']
    sample_query = sample_dict_LO['query']
    sample_cot = sample_dict_LO['cot']
    sample_prop_vars = sample_dict_LO['prop_vars']
    sample_context_LO, sample_answer_LO, _ = generate_one_sample(sample_rules, sample_facts, sample_query, sample_cot)
    output_string_LO += "" + convert_to_english(sample_context_LO)[1] + " "
    output_string_LO += "Answer:"

    queried_rule_LO = sample_dict_LO['queried_rule']
    correct_fact_LO = sample_dict_LO['correct_fact']
    queried_rule_LO = convert_to_english(queried_rule_LO, is_question=False)[1]
    correct_fact_LO = convert_to_english(correct_fact_LO, is_question=False)[1]

    # Now the LINEAR string
    sample_rules = sample_dict_lin['rules']
    sample_facts = sample_dict_lin['facts']
    sample_query = sample_dict_lin['query']
    sample_cot = sample_dict_lin['cot']
    sample_prop_vars = sample_dict_lin['prop_vars']
    sample_context_lin, sample_answer_lin, _ = generate_one_sample(sample_rules, sample_facts, sample_query, sample_cot)
    output_string_lin += "" + convert_to_english(sample_context_lin)[1] + " "
    output_string_lin += "Answer:"

    queried_rule_lin = sample_dict_lin['queried_rule']
    correct_fact_lin = sample_dict_lin['correct_fact']
    
    queried_rule_lin = convert_to_english(queried_rule_lin, is_question=False)[1]
    correct_fact_lin = convert_to_english(correct_fact_lin, is_question=False)[1]

    problem_info_dict_LO = {'queried_rule': queried_rule_LO,
                            'correct_fact': correct_fact_LO}
    problem_info_dict_lin = {'queried_rule': queried_rule_lin,
                             'correct_fact': correct_fact_lin}

    gt_string_LO += convert_to_english(sample_answer_LO, is_question=False)[1]
    gt_string_lin += convert_to_english(sample_answer_lin, is_question=False)[1]


    
  return output_string_LO, output_string_lin, gt_string_LO, gt_string_lin, problem_info_dict_LO, problem_info_dict_lin

def generate_cot_question_query_based(length_of_chain = 2, num_cot_samples = 6):

  EXAMPLE_STRING, _ = sample_context_and_answer_pairs_EXAMPLES(num_cot_samples, length_of_chain, print_answer = True, randomness=False)

  NEW_STRING_LO, NEW_STRING_LIN, gt_string_LO, gt_string_lin, problem_info_dict_LO, problem_info_dict_lin = sample_context_and_answer_pairs_QUESTION(1, length_of_chain, print_answer = False, randomness=False)

  full_string_LO = EXAMPLE_STRING + NEW_STRING_LO
  full_string_linear = EXAMPLE_STRING + NEW_STRING_LIN

  return full_string_LO, gt_string_LO, full_string_linear, gt_string_lin, problem_info_dict_LO, problem_info_dict_lin
