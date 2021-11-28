import random
import argparse
import time
import json
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from scipy.special import softmax

topics_list = ['weed', 'abortion']

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

    def get_opinion(self, user_query, previous_belief):
        inputs = self.tokenizer([user_query], return_tensors='pt')
        output = self.model(**inputs)
        results = softmax(output[0][0].detach().numpy())

        # 0 - negative 1 - neutral 2 - positive
         

def socratic_chatbot(current_topic, topic_idx, user_name):
    json_file = current_topic + '.json'
    
    with open('./FSM/'+json_file) as f:
            fsm = json.load(f)

    opinion_classifier = opinionClassifier()
    context = []
    current_state = 'state_0'
    current_belief = ''
    while True:
        state = fsm[current_state]
        print('Mark : ' + state['response'])
        response = input(user_name + " : ")
        current_belief = opinion_classifier.get_belief(response, current_belief)
        current_state = state['next_state'][current_belief]

def normal_chatbot(current_topic, topic_idx, user_name):
    json_file = current_topic + '.json'

    with open('./FSM/'+json_file) as f:
            fsm = json.load(f)
    
    opinion_classifier = opinionClassifier()
    turn = 0
    context = []
    while turn < 2:
        print('Mark : Hey champ!!! How was the day?')
        response = input(user_name + " : ")
        turn += 1

    # response to move away from topic
    diverging_sentence = fsm['diverging statement']
    print('Mark : ' + diverging_sentence)
    response = input(user_name + " : ")
    context.append(diverging_sentence)
    context.append(response)
    turn += 1

    # start using an open-domain dialog model for response generation
    blender_bot = blenderBot()

    while turn < 8:
        model_response = blender_bot.get_response(" ".join(context))
        user_response = input(user_name + " : ")
        context.append(model_response)
        context.append(user_response)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='chatbot for socratc questioning')
    parser.add_argument('--group_type', type=int, default=0, metavar='N', help='conditional group(0 for socratic chatbot, 1 for normal chatbot)')
    parser.add_argument('--name', type=str, default='', help='get name of the participant')
    args = parser.parse_args()
    
    topic_idx = random.randint(0, len(topics_list)-1)
    current_topic = topics_list[topic_idx]

    # Check name of the participant
    if args.name == '':
        print("Please enter our friend's name!")
        sys. exit()
    else:
        user_name = args.name

    # Check the group type
    if args.group_type == 0:
        socratic_chatbot(current_topic, topic_idx, user_name)
    elif args.group_type == 1:
        normal_chatbot(current_topic, topic_idx, user_name)
    else:
        print('Please type correct group type!!!')
    
