#!/usr/bin/env python

import json
import ast
import argparse


def get_conversation_round(idx: int, data: str, final_dict: dict):
    data_dict = ast.literal_eval(data)
    # {'round-1': {'user': 'Any advice on starting a meditation practice?', 'AI': 'Begin with short sessions, around 5-10 minutes. Focus on your breath and gradually increase the duration as you become more comfortable.', 'interruption': 'Further Inquiry'}, 'round-2': {'user': 'Can meditation help with stress relief?', 'AI': 'Absolutely, meditation can significantly reduce stress by promoting relaxation and mindfulness, helping you to focus better and feel calmer.', 'interruption': 'Further Inquiry'}, 'round-3': {'user': "Oh, also, what's a good app for guided meditation?", 'AI': '"Headspace" and "Calm" are popular apps with guided meditations for various needs, like stress reduction and sleep improvement.', 'interruption': 'Topic Shift'}, 'round-4': {'user': 'By the way, do you know if I should meditate in the morning or evening?', 'AI': 'It depends on personal preference. Mornings can boost focus for the day, while evenings help unwind and relax before sleep.', 'interruption': 'Further Inquiry'}}
    temp_data = dict()
    round_num = len(data_dict)
    temp_data["num_rounds"] = round_num
    user_string = ""
    ai_string = ""
    for round, (_, value) in enumerate(data_dict.items()):
        # Get user strings first
        # "Any advice on starting a meditation practice? <Further Inquiry> Can meditation help with stress relief? <Further Inquiry> Oh, also, what's a good app for guided meditation? <Topic Shift> By the way, do you know if I should meditate in the morning or evening? "
        user_string = temp_data.get("user", "")
        user_string += value["user"] + " "
        if round < round_num - 1: 
            # user_string += "<" + value["interruption"] + "> "
            user_string += "| "
        
        # Get AI strings then
        # "Begin with short sessions, around 5-10 minutes. Focus on your breath and gradually increase the duration as you become more comfortable. <Further Inquiry> Absolutely, meditation can significantly reduce stress by promoting relaxation and mindfulness, helping you to focus better and feel calmer. <Further Inquiry> "Headspace" and "Calm" are popular apps with guided meditations for various needs, like stress reduction and sleep improvement. <Topic Shift> It depends on personal preference. Mornings can boost focus for the day, while evenings help unwind and relax before sleep. "
        ai_string = temp_data.get("ai", "")
        ai_string += value["AI"] + " "
        if round < round_num - 1: 
            # ai_string += "<" + value["interruption"] + "> "
            ai_string += "| "
        
        temp_data["user"] = user_string
        temp_data["ai"] = ai_string
    final_dict[idx+1] = temp_data


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_json", type=str, default="./data/output_all_sort.json")
    argparser.add_argument("--output_json", type=str, default="./data/conversation_round_no_interruption_label.json")
    args = argparser.parse_args()
    final_dict = dict()
    with open(args.input_json, "r") as f:
        for idx, line in enumerate(f):
            get_conversation_round(idx, line.replace("\n", ""), final_dict)
            
    # print(final_dict)
    with open(args.output_json, "w") as f:
        json.dump(final_dict, f, indent=4, ensure_ascii=False)
    
    
if __name__ == "__main__":
    main()