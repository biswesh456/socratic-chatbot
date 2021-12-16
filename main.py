import random
import argparse
import time
import json
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from scipy.special import softmax
from datetime import datetime

topics_list = ['weed', 'gene']
log = []
topic_changed = False

class blenderBot:
    def __init__(self):
        """Initialize model and tokenizer."""
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-1B-distill")

    def get_response(self, user_query):
        inputs = self.tokenizer([user_query], return_tensors='pt')
        reply_ids = self.model.generate(**inputs)
        bot_answer = self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
        return bot_answer

class opinionClassifier:
    def __init__(self):
        """Initialize model and tokenizer."""
        self.model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
        self.tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")

    def get_opinion(self, user_query):
        user_query = user_query.lower()
        user_query = user_query.replace('favour', 'for')

        if "not sure" in user_query or "dont know" in user_query or "don't know" in user_query:
            return ["neutral", "against"]

        if "not against" in user_query or "not in against" in user_query or "not in the against" in user_query:
            return ["for"]

        if "not for" in user_query or "not in for" in user_query or "not in the for" in user_query:
            return ["against"]

        inputs = self.tokenizer([user_query], return_tensors='pt')
        output = self.model(**inputs)
        results = softmax(output[0][0].detach().numpy())

        # 0 - negative 1 - neutral 2 - positive
        if results[0] > 0.5:
            return ["against"]
        elif results[2] > 0.25 and results[0] < 0.1:
            return ["for"]
        else:
            if results[0] > 0.25:
                return ["neutral", "against"]
            else:
                if 'for' in user_query:
                    return ["for"]
                return ["neutral", "for"]

    def yes_no_detector(self, user_query):
        user_query = user_query.lower()
        if any(x in ["no ", "nope", "not ", "don't"] for x in user_query.split()):
            return "no"
        else:
            return "yes"

def detect_reason(response, prev_user_response, p, q):
    prev_user_response = prev_user_response.replace('I ', 'you ')
    prev_user_response = prev_user_response.replace(' am ', ' are ')
    prev_user_response = prev_user_response.replace(' was ', ' were ')
    prev_user_response = prev_user_response.lower()
    
    if '[P]' in response and '[Q]' not in response:
        if 'because' in prev_user_response:
            p = prev_user_response.split('because')[-1]
            p = p.replace('!', '').replace('?', '')
            return p, '', response.replace('[P]', p).strip() 
        
        elif 'since' in prev_user_response:
            p = prev_user_response.split('since')[-1]
            p = p.replace('!', '').replace('?', '')
            return p, '', response.replace('[P]', p).strip() 

        elif ' as ' in prev_user_response:
            p = prev_user_response.split(' as ')[-1]
            p = p.replace('!', '').replace('?', '')
            return p, '', response.replace('[P]', p).strip() 

        elif 'that' in prev_user_response:
            p = prev_user_response.split('that')[-1]
            p = p.replace('!', '').replace('?', '')
            return p, '', response.replace('[P]', p).strip() 
        else:
            p = prev_user_response
            p = p.replace('!', '').replace('?', '')
            return p, '', response.replace('[P]', p).strip() 

    elif '[Q]' in response:
        if 'because' in prev_user_response:
            q = prev_user_response.split('because')[-1]
            q = q.replace('!', '').replace('?', '')
            return p, q, response.replace('[P]', p).replace('[Q]', q).strip() 
        
        elif 'since' in prev_user_response:
            q = prev_user_response.split('since')[-1]
            q = q.replace('!', '').replace('?', '')
            return p, q, response.replace('[P]', p).replace('[Q]', q).strip() 

        elif ' as ' in prev_user_response:
            q = prev_user_response.split(' as ')[-1]
            q = q.replace('!', '').replace('?', '')
            return p, q, response.replace('[P]', p).replace('[Q]', q).strip() 

        elif 'that' in prev_user_response:
            q = prev_user_response.split('that')[-1]
            q = q.replace('!', '').replace('?', '')
            return p, q, response.replace('[P]', p).replace('[Q]', q).strip() 
        else:
            q = prev_user_response
            return p, q, response.replace('[P]', p).replace('[Q]', q).strip()

    else:
        return p, q, response 
            
        

def socratic_chatbot(current_topic, user_name, current_state='state_0'):
    global topic_changed
    json_file = current_topic + '.json'
    
    with open('./FSM/'+json_file) as f:
            fsm = json.load(f)

    opinion_classifier = opinionClassifier()
    context = []
    current_belief = ''
    reason_p = ''
    reason_q = ''
    user_response = ''

    while True:
        state = fsm[current_state]
        response = state['Response'].replace("[NAME]", user_name)
        reason_p, reason_q, response = detect_reason(response, user_response, reason_p, reason_q)

        time.sleep(len(response.split())*0.2)
        
        now = datetime.now()
        start_time = now.strftime("%H:%M:%S")
        print('Mark : ' + response)
        user_response = input(user_name + " : ")
        now = datetime.now()
        stop_time = now.strftime("%H:%M:%S")

        log.append(['Mark : ' + response, user_name + ' : ' + user_response, str(start_time), str(stop_time)])

        if 'Next_State' in state:
            next_state = state['Next_State']
            if isinstance(next_state, str):
                if "change" in next_state:
                    topic_changed = True
                    current_state = next_state[7:]
                    
                    if current_topic == "weed":
                        current_topic = "gene"
                    else:
                        current_topic = "weed"
                    
                    socratic_chatbot(current_topic, user_name, current_state)
                    break
                else:
                    current_state = next_state
            else:
                # If next state depends on opinion
                if "for" in next_state:
                    current_belief = opinion_classifier.get_opinion(user_response)
                    if current_belief[0] == "for":
                        current_state = next_state["for"]
                    elif current_belief[0] == "against":
                        current_state = next_state["against"]
                    else:
                        if "neutral" in next_state:
                            if topic_changed == True:
                                current_state = next_state[current_belief[1]]
                            else:    
                                current_state = next_state["neutral"]
                        else:
                            current_state = next_state[current_belief[1]]

                # If next state depends upon yes or no type of answer
                elif "yes" in next_state:
                    current_belief = opinion_classifier.yes_no_detector(user_response)
                    current_state = next_state[current_belief]
                else:
                    print("Some Error!!!")
                    break
        else:
            # End of FSM
            break

def normal_chatbot(current_topic, user_name, current_state='state_0'):
    # start using an open-domain dialog model for response generation
    blender_bot = blenderBot()
    json_file = current_topic + '.json'

    with open('./FSM/'+json_file) as f:
            fsm = json.load(f)
    
    opinion_classifier = opinionClassifier()
    turn = 0
    context = []
    while True:
        state = fsm[current_state]
        response = state['Response'].replace("[NAME]", user_name)
        time.sleep(len(response.split())*0.2)

        now = datetime.now()
        start_time = now.strftime("%H:%M:%S")
        print('Mark : ' + response)
        user_response = input(user_name + " : ")
        now = datetime.now()
        stop_time = now.strftime("%H:%M:%S")
        log.append(['Mark : ' + response, user_name + ' : ' + user_response, str(start_time), str(stop_time)])

        if current_state == "state_3":
            break
        
        current_state = state["Next_State"]

    # response to move away from topic
    diverging_sentence = fsm['state_diverging_statement']["Response"]
    now = datetime.now()
    start_time = now.strftime("%H:%M:%S")
    print('Mark : ' + diverging_sentence)
    response = input(user_name + " : ")
    now = datetime.now()
    stop_time = now.strftime("%H:%M:%S")

    context.append(diverging_sentence)
    context.append(response)

    log.append(['Mark : ' + diverging_sentence, user_name + ' : ' + response, str(start_time), str(stop_time)])

    turn += 1

    while turn < 7:
        model_response = blender_bot.get_response(" ".join(context))
        now = datetime.now()
        start_time = now.strftime("%H:%M:%S")
        print('Mark : ' + model_response)
        user_response = input(user_name + " : ")
        now = datetime.now()
        stop_time = now.strftime("%H:%M:%S")

        context.append(model_response)
        context.append(user_response)

        log.append(['Mark : ' + response, user_name + ' : ' + user_response, str(start_time), str(stop_time)])

    
    print('Mark : Anyways, it was nice to talk to you! I got to go now. See you soon... Bye!!')
    log.append('Mark : Anyways, it was nice to talk to you! I got to go now. See you soon... Bye!!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='chatbot for socratc questioning')
    parser.add_argument('--group', type=int, default=0, metavar='N', help='conditional group(0 for socratic chatbot, 1 for normal chatbot)')
    parser.add_argument('--name', type=str, default='', help='get name of the participant')
    args = parser.parse_args()
    
    topic_idx = random.randint(0, len(topics_list)-1)
    current_topic = topics_list.pop(topic_idx)

    # Check name of the participant
    if args.name == '':
        print("Please enter our friend's name!")
        sys. exit()
    else:
        user_name = args.name

    try:
        # Check the group type
        if args.group == 0:
            socratic_chatbot(current_topic, user_name)
        elif args.group == 1:
            normal_chatbot(current_topic, user_name)
        else:
            print('Please type correct group type!!!')

        # add log
        with open('log/' + user_name + '_' + str(args.group) + '.json', 'w') as f:
            json.dump(log, f)

    except KeyboardInterrupt:
        with open('log/' + user_name + '_' + str(args.group) + '.json', 'w') as f:
            json.dump(log, f)
