#!/usr/bin/env python

# import json
import torch
import numpy as np
import random
import ast
import os
from typing import List
import matplotlib.pyplot as plt
import whisperx
import sys, ipdb
import subprocess


def count_greater_than_zero(lst: List):
    """
    Recursively counts the number of elements greater than zero in a nested list structure.
    
    Args:
        lst (List): A list that may contain numbers or other nested lists.
        
    Returns:
        int: The total count of numeric elements greater than zero across all nesting levels.
        
    Example:
        >>> count_greater_than_zero([1, 0, [2, -1, [3, 0]], -5])
        3
    """
    count = 0
    for item in lst:
        if isinstance(item, list):  # Check if the item is a list
            count += count_greater_than_zero(item)  # Recursively count in the sublist
        elif isinstance(item, (int, float)) and item > 0:  # Check if item is a number greater than 0
            count += 1
    return count


def flatten_list(lst):
    """
    Recursively flattens a nested list structure into a single-level list.
    
    Args:
        lst: A list that may contain nested lists at any depth.
        
    Returns:
        list: A flattened list containing all non-list elements from the input.
        
    Example:
        >>> flatten_list([1, [2, 3], [4, [5, 6]]])
        [1, 2, 3, 4, 5, 6]
    """
    flattened = []
    for item in lst:
        if isinstance(item, list):  # Check if the item is a list
            flattened.extend(flatten_list(item))  # Recursively flatten the sublist
        else:
            flattened.append(item)  # Append non-list items directly
    return flattened


def time_average(lst):
    # """
    # Computes the average of a list of time intervals in milliseconds.

    # Parameters:
    # - lst (list): A list of time intervals in milliseconds.

    # Returns:
    # - float: The average time interval in milliseconds.
    # """
    # if len(lst) == 0:
    #     return 0
    # return sum(lst) / len(lst)
    """
    Computes the median of a list of time intervals in milliseconds.

    Parameters:
    - lst (list): A list of time intervals in milliseconds.

    Returns:
    - float: The median time interval in milliseconds.
    """
    if not lst:
        return 0

    # Sort the list to arrange the numbers in order.
    sorted_lst = sorted(lst)
    n = len(sorted_lst)
    mid = n // 2

    # If the number of elements is odd, return the middle element.
    if n % 2 == 1:
        return sorted_lst[mid]
    else:
        # If even, return the average of the two middle values.
        return (sorted_lst[mid - 1] + sorted_lst[mid]) / 2


def draw_pie_chart_delay(ms_list, title, output_file='output.png'):
    """
    Draws a pie chart based on the given list of milliseconds and intervals.

    Parameters:
    - ms_list (list): A list of millisecond values.
    - title (list): Title for the chart

    Returns:
    None
    """
    
    # Initialize counts for each interval
    intervals = [(0, 100), (100, 200), (200, 500), (500, 1000), (1000, 3000), (3000, float('inf'))]
    labels = ['0-100ms', '100-200ms', '200-500ms', '500-1000ms', '1000-3000ms', '3000ms+']
    counts = [0] * len(intervals)

    # Count occurrences in each interval
    for ms in ms_list:
        for i, (low, high) in enumerate(intervals):
            if low <= ms < high:
                counts[i] += 1
                break

    # Plot the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    # plt.show()
    plt.savefig(output_file, format='png', dpi=300)
    

def draw_pie_chart_lead(ms_list, title, output_file='output.png'):
    """
    Draws a pie chart based on the given list of milliseconds and intervals.

    Parameters:
    - ms_list (list): A list of millisecond values.
    - title (list): Title for the chart

    Returns:
    None
    """
    
    # Initialize counts for each interval
    intervals = [(0, 100), (100, 200), (200, 500), (500, 1000)]
    labels = ['0-100ms', '100-200ms', '200-500ms', '500-1000ms']
    counts = [0] * len(intervals)

    # Count occurrences in each interval
    for ms in ms_list:
        for i, (low, high) in enumerate(intervals):
            if low <= ms < high:
                counts[i] += 1
                break

    # Plot the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    # plt.show()
    plt.savefig(output_file, format='png', dpi=300)


def find_ai_speaks_idx(ai_time_starts, user_time_starts):
    """
    For each time in ai_time_starts, find the index i in user_time_starts
    such that user_time_starts[i] < ai_time < user_time_starts[i+1].
    If ai_time is greater than or equal to the last element in user_time_starts,
    assign the index of the last element.
    
    Parameters:
        ai_time_starts (list of numbers): List of AI time stamps.
        user_time_starts (list of numbers): List of user time stamps (assumed sorted).
    
    Returns:
        list of int: The indices corresponding to the window each AI time stamp falls into.
    """
    indices = []
    for ai_time in ai_time_starts:
        found = False
        # Loop through intervals defined by consecutive user time stamps
        for idx in range(len(user_time_starts) - 1):
            if user_time_starts[idx] < ai_time < user_time_starts[idx+1]:
                indices.append(idx)
                found = True
                break
        if not found:
            # If ai_time is greater than or equal to the last user time stamp, assign last index.
            if ai_time >= user_time_starts[-1]:
                indices.append(len(user_time_starts) - 1)
            else:
                # Optionally handle the case where ai_time is less than the first user time stamp.
                indices.append(0)
    return indices

class Benchmarking():
    def __init__(self, input_data: str, ground_truth: str, interruption: str, model_name: str = "Moshi"):
        """
        Initialize a Benchmarking instance for evaluating conversational AI models.
        
        Args:
            input_data (str): Path to the input data file containing conversation results.
            ground_truth (str): Path to the ground truth file containing reference conversations.
            interruption (str): Type of interruption analysis to perform.
            model_name (str, optional): Name of the model being evaluated. Defaults to "Moshi".
            
        Sets up various tracking dictionaries and metrics for analyzing conversation quality,
        interruption handling, response timing, and speech recognition accuracy.
        """
        # {'num_rounds': 1382, 'num_interruptions': 1196, 'interruption_types': {'Further Inquiry': 378, 'Topic Shift': 301, 'Affirmative Acknowledgment': 321, 'Third-Party Noise': 109, 'Denial and Discontent': 57, 'Affirmative Acknowledgment + Topic Shift': 14, 'Affirmative Acknowledgment + Further Inquiry': 15, 'Denial and Discontent + Topic Shift': 1}}
        self.interruption_cats = {"Further Inquiry": 0, "Topic Shift": 0, "Affirmative Acknowledgment": 0, "Third-Party Noise": 0, "Denial and Discontent": 0}
        self.interruption_success_cats = {"Further Inquiry": 0, "Topic Shift": 0, "Affirmative Acknowledgment": 0, "Third-Party Noise": 0, "Denial and Discontent": 0}
        self.interruption_res_suc_cats = {"Further Inquiry": 0, "Topic Shift": 0, "Affirmative Acknowledgment": 0, "Third-Party Noise": 0, "Denial and Discontent": 0}
        self.metrics = dict()
        self.debug = True
        self.strict = True
        self.model_name = model_name
        self.interruption_type = interruption
        self.speech_recognition = "whisper-large-v3-turbo"
        self.asr_compute_type = "float16"
        self.asr_device = "cuda"
        self.asr_batch_size = 16
        self.asr_model = None
        self.hyp = None
        self.ref = None
        self.wer = None
        self.input_data = input_data
        self.VAD_marks = dict()
        self.all_data = open(self.input_data, 'r').readlines()
        self.ground_truth = open(ground_truth, 'r').readlines()
        self.ground_truth_dict = dict()
        self.quality_check_dict = dict()
        self.conv_id = []
        self.response_token_id = []
        self.response_text = []
        self.response_text_segments = dict()
        self.user_timestamps = dict()
        self.ai_timestamps = dict()
        print("processing {}".format(self.input_data))
    # The input 
    # Based on VAD results, we will first compute             
    # InterruptSuccessRate ISR(%) = SuccessInterruptCount / UserInterruptCount.
    # conversation_177.wav || [3, 3, 3, 3, 0, ..., 5287, 3, 3, 3] || [' Hello', ',', ' how', ' can', ' I', ' help', ' you', '?', ... ' Okay', '.', ' Sure', ',', ' what', ' do', ' I', ' need', '?', ' Sure', ',', ' I', ' recommend', ' starting', ' with', ' a', ' basic', ' photography', ' course', ' online', ' or', ' in', ' person', '.', ' will', ' give', ' you', ' a', ' good', ' foundation'] || [{'start': 53280, 'end': 93664}, {'start': 218656, 'end': 250848}, {'start': 397856, 'end': 422368}, {'start': 565280, 'end': 597472}, {'start': 752160, 'end': 794592}] || [{'start': 7200, 'end': 36320}, {'start': 78368, 'end': 83936}, {'start': 142368, 'end': 322528}, {'start': 344096, 'end': 488928}, {'start': 573984, 'end': 655840}, {'start': 770592, 'end': 804320}, {'start': 848416, 'end': 904160}, {'start': 935456, 'end': 953600}]

    def count_user_interruptions(self, interruption_type: str):
        """
        Increment counters for different types of user interruptions.
        
        Args:
            interruption_type (str): The type of interruption (e.g., "Further Inquiry", 
                "Topic Shift", "Affirmative Acknowledgment", etc.)
                
        Returns:
            str: The interruption type that was processed.
            
        Handles both single interruption types and combined types (e.g., 
        "Affirmative Acknowledgment + Topic Shift").
        """
        if interruption_type == "Further Inquiry":
            self.interruption_cats["Further Inquiry"] += 1
        elif interruption_type == "Topic Shift":
            self.interruption_cats["Topic Shift"] += 1
        elif interruption_type == "Affirmative Acknowledgment":
            self.interruption_cats["Affirmative Acknowledgment"] += 1
        elif interruption_type == "Third-Party Noise":
            self.interruption_cats["Third-Party Noise"] += 1
        elif interruption_type == "Denial and Discontent":
            self.interruption_cats["Denial and Discontent"] += 1
        elif interruption_type == "Affirmative Acknowledgment + Topic Shift":
            self.interruption_cats["Affirmative Acknowledgment"] += 1
            self.interruption_cats["Topic Shift"] += 1
        elif interruption_type == "Affirmative Acknowledgment + Further Inquiry":
            self.interruption_cats["Affirmative Acknowledgment"] += 1
            self.interruption_cats["Further Inquiry"] += 1
        elif interruption_type == "Denial and Discontent + Topic Shift":
            self.interruption_cats["Denial and Discontent"] += 1
            self.interruption_cats["Topic Shift"] += 1
            
        return interruption_type
    
    def count_user_interruptions_success(self, interruption_type: str):
        """
        Increment counters for successful user interruptions by type.
        
        Args:
            interruption_type (str): The type of interruption that was successful.
            
        Tracks how many interruptions of each type were successfully handled by the AI,
        supporting both single and combined interruption types.
        """
        if interruption_type == "Further Inquiry":
            self.interruption_success_cats["Further Inquiry"] += 1
        elif interruption_type == "Topic Shift":
            self.interruption_success_cats["Topic Shift"] += 1
        elif interruption_type == "Affirmative Acknowledgment":
            self.interruption_success_cats["Affirmative Acknowledgment"] += 1
        elif interruption_type == "Third-Party Noise":
            self.interruption_success_cats["Third-Party Noise"] += 1
        elif interruption_type == "Denial and Discontent":
            self.interruption_success_cats["Denial and Discontent"] += 1
        elif interruption_type == "Affirmative Acknowledgment + Topic Shift":
            self.interruption_success_cats["Affirmative Acknowledgment"] += 1
            self.interruption_success_cats["Topic Shift"] += 1
        elif interruption_type == "Affirmative Acknowledgment + Further Inquiry":
            self.interruption_success_cats["Affirmative Acknowledgment"] += 1
            self.interruption_success_cats["Further Inquiry"] += 1
        elif interruption_type == "Denial and Discontent + Topic Shift":
            self.interruption_success_cats["Denial and Discontent"] += 1
            self.interruption_success_cats["Topic Shift"] += 1

    def count_user_interruptions_response_success(self, interruption_type: str):
        """
        Increment counters for successful AI responses to user interruptions by type.
        
        Args:
            interruption_type (str): The type of interruption that received a successful response.
            
        Tracks how many interruptions of each type received appropriate responses from the AI,
        supporting both single and combined interruption types.
        """
        if interruption_type == "Further Inquiry":
            self.interruption_res_suc_cats["Further Inquiry"] += 1
        elif interruption_type == "Topic Shift":
            self.interruption_res_suc_cats["Topic Shift"] += 1
        elif interruption_type == "Affirmative Acknowledgment":
            self.interruption_res_suc_cats["Affirmative Acknowledgment"] += 1
        elif interruption_type == "Third-Party Noise":
            self.interruption_res_suc_cats["Third-Party Noise"] += 1
        elif interruption_type == "Denial and Discontent":
            self.interruption_res_suc_cats["Denial and Discontent"] += 1
        elif interruption_type == "Affirmative Acknowledgment + Topic Shift":
            self.interruption_res_suc_cats["Affirmative Acknowledgment"] += 1
            self.interruption_res_suc_cats["Topic Shift"] += 1
        elif interruption_type == "Affirmative Acknowledgment + Further Inquiry":
            self.interruption_res_suc_cats["Affirmative Acknowledgment"] += 1
            self.interruption_res_suc_cats["Further Inquiry"] += 1
        elif interruption_type == "Denial and Discontent + Topic Shift":
            self.interruption_res_suc_cats["Denial and Discontent"] += 1
            self.interruption_res_suc_cats["Topic Shift"] += 1
    
    def get_user_inquiry(self):
        """
        Parse and organize user conversation data from ground truth.
        
        Processes ground truth conversation data to create a structured dictionary
        containing user inquiries, interruption types, and timestamps. Maps user
        utterances to conversation rounds and stores the interruption type for
        the next round.
        
        Updates:
            self.ground_truth_dict: Dictionary mapping conversation IDs to round data
            containing user text, interruption type, and timestamps.
        """
        for idx, conv in enumerate(self.ground_truth):
            name = "conversation_" + str(idx + 1) + ".wav"
            user_timestamps = self.user_timestamps.get(name)
            conversation = ast.literal_eval(conv)
            try:
                assert len(user_timestamps) == len(conversation), "User Timestamps do not match the conversation length, {} vs {}".format(len(user_timestamps), len(conversation))
            except TypeError:
                continue
            new_dict = dict()
            next_interruption = None
            for key, value in conversation.items():
                if key == "round-1":
                    new_dict[1] = {"User": value.get("user"), "Type": "First Inquiry", "Timestamps": user_timestamps[0]}
                    next_interruption = value.get("interruption")
                elif key == "round-{}".format(str(len(conversation))):
                    new_dict[int(key.split("-")[1])] = {"User": value.get("user"), "Type": next_interruption, "Timestamps": user_timestamps[int(key.split("-")[1]) - 1]}
                    next_interruption = "Done" # Never used since this is already the last round 
                else:
                    new_dict[int(key.split("-")[1])] = {"User": value.get("user"), "Type": next_interruption, "Timestamps": user_timestamps[int(key.split("-")[1]) - 1]}
                    next_interruption = value.get("interruption")
            self.ground_truth_dict[name] = new_dict
            
    
    def analyze_VAD_interruption_new(self):
        """
        Analyze Voice Activity Detection (VAD) data to evaluate interruption handling.
        
        This method performs detailed analysis of user-AI conversation timing to determine:
        - Success rates for AI responses to user inputs
        - Timing metrics for interruptions and responses
        - Classification of different types of interruptions (wrong, noise, successful)
        
        The analysis uses user timestamps as the basis and evaluates AI behavior patterns:
        - AI responses that start during user speech (potential interruptions)
        - Response delays and lead times
        - Successful vs unsuccessful interruption handling
        
        For Moshi model, discards the first AI segment if it's a preamble.
        
        Updates:
            self.VAD_marks: Dictionary containing detailed analysis results for each conversation
            including success rates, timing metrics, and categorized interruption data.
        """
        # Rewritten. Using User's start/end as the basis.
        # For the AI part, if there is a preamble, discard the first one (moshi).
        # Recurrently go through the AI's start/end:
        #   If the start falls between the user's start-end, it might be a wrong interruption.
        #       If the gap between start to user's end is less than 1 seconds, it is a correct response, calculate the lead-time.
        #       If the gap between start to user's end is more than 1 seconds, it is a wrong interruption.
        #   If the start falls between the user's end-start, it is a correct response, calculate the delay.
        #   If the end falls between the user's start-end, it is a correct interruption, calculate the delay.
        #   If the end falls between the user's end-start, then determine the distance between the end and the user's start, 
        #      If it exceeds 2.5s, it indicates an interruption by a third party (currently only the last segment is implemented as noise data, additional noise interruptions need to be generated).
        self.get_user_inquiry()
        for conv_id, user_timestamp in self.user_timestamps.items():
            ground_truth = self.ground_truth_dict.get(conv_id)
            ai_timestamp = self.ai_timestamps.get(conv_id)
            ai_starts = [a['start'] for a in ai_timestamp]
            ai_ends = [a['end'] for a in ai_timestamp]
            user_starts = [u['start'] for u in user_timestamp]
            user_ends = [u['end'] for u in user_timestamp]
            
            total_conversation_rounds = len(user_timestamp)
            total_interruptions = total_conversation_rounds - 1
            total_real_interruptions = 0
            
            count_wrong_interrupt_ai = 0
            lead_time_interrupt = [0 for _ in range(total_conversation_rounds)] # lead for wrong interrupt
            
            count_noise_interrupt = 0
            
            success_response = 0
            success_response_to_interruption = 0
            response_delay = [0 for _ in range(total_conversation_rounds)] # delay for successful response
            response_delay_to_interruption = [0 for _ in range(total_conversation_rounds)] # delay for successful response to interruption
            lead_times = [0 for _ in range(total_conversation_rounds)]
            lead_time_to_interruption = [0 for _ in range(total_conversation_rounds)]
            
            success_interrupt = 0
            interrupt_delay = [0 for _ in range(total_interruptions)]  # delay for successful interrupt
            
            
            
            if self.model_name == "Moshi":
                # Discard the first AI start/end if there is a preamble
                ai_starts = ai_starts[1:]
                ai_ends = ai_ends[1:]
                

            # We first check AI starts
            current_user_interrupt = False
            for a_i, ai_start in enumerate(ai_starts):
                for i, u_start in enumerate(user_starts):
                    u_end = user_ends[i]
                    # current_user_interrupt = False
                    if ai_start < u_start < ai_ends[a_i]: # (ai_start > u_start and ai_ends[a_i] < u_end)
                        total_real_interruptions += 1
                        cur_interruption_type = self.count_user_interruptions(ground_truth.get(i + 1).get("Type"))
                        current_user_interrupt = True
                    if u_start < ai_start < u_end - 8000:
                        count_wrong_interrupt_ai += 1
                        if lead_time_interrupt[i] == 0:
                            lead_time_interrupt[i] = u_end - ai_start
                        elif type(lead_time_interrupt[i]) == int:
                            current_lead = [lead_time_interrupt[i], u_end - ai_start]
                            lead_time_interrupt[i] = current_lead
                        elif type(lead_time_interrupt[i]) == list:
                            current_lead = lead_time_interrupt[i]
                            current_lead.append(u_end - ai_start)
                            lead_time_interrupt[i] = current_lead
                            
                        if ai_ends[a_i] < u_end:
                            total_real_interruptions += 1
                            cur_interruption_type = self.count_user_interruptions(ground_truth.get(i + 1).get("Type"))
                            success_interrupt += 1
                            current_user_interrupt = True
                            try: 
                                if interrupt_delay[i - 1] == 0:
                                    interrupt_delay[i - 1] = ai_ends[a_i] - ai_start
                                    self.count_user_interruptions_success(cur_interruption_type)
                                elif type(interrupt_delay[i - 1]) == int:
                                    current_delay = [interrupt_delay[i - 1], ai_ends[a_i] - ai_start]
                                    interrupt_delay[i - 1] = current_delay
                                    self.count_user_interruptions_success(cur_interruption_type)
                                elif type(interrupt_delay[i - 1]) == list:
                                    current_delay = interrupt_delay[i - 1]
                                    current_delay.append(ai_ends[a_i] - ai_start)
                                    interrupt_delay[i - 1] = current_delay
                                    self.count_user_interruptions_success(cur_interruption_type)
                            # refresh the input using timestamp of AI response
                            except IndexError:
                                print("Conversation Error: ", conv_id)
                        continue
                    elif u_start < ai_start < u_end:
                        if ai_ends[a_i] < u_end:
                            success_interrupt += 1
                            try: 
                                if interrupt_delay[i - 1] == 0:
                                    interrupt_delay[i - 1] = ai_ends[a_i] - ai_start
                                    self.count_user_interruptions_success(cur_interruption_type)
                                elif type(interrupt_delay[i - 1]) == int:
                                    current_delay = [interrupt_delay[i - 1], ai_ends[a_i] - ai_start]
                                    interrupt_delay[i - 1] = current_delay
                                    self.count_user_interruptions_success(cur_interruption_type)
                                elif type(interrupt_delay[i - 1]) == list:
                                    current_delay = interrupt_delay[i - 1]
                                    current_delay.append(ai_ends[a_i] - ai_start)
                                    interrupt_delay[i - 1] = current_delay
                                    self.count_user_interruptions_success(cur_interruption_type)
                            # refresh the input using timestamp of AI response
                            except IndexError:
                                print("Conversation Error: ", conv_id)
                        if not current_user_interrupt:
                            success_response += 1
                            lead_times[i] = u_end - ai_start
                        else: 
                            success_response_to_interruption += 1
                            self.count_user_interruptions_response_success(cur_interruption_type)
                            lead_time_to_interruption[i] = u_end - ai_start
                            current_user_interrupt = False
                        continue
                    
                    
                    
                    if i < len(user_starts) - 1:
                        next_u_start = user_starts[i+1]
                        if u_end < ai_ends[a_i] < next_u_start:
                            if current_user_interrupt:
                                success_interrupt += 1
                                self.count_user_interruptions_success(cur_interruption_type)
                                current_user_interrupt = False
                            continue
                        if u_end < ai_start < next_u_start:
                            if not current_user_interrupt:
                                success_response += 1 if response_delay[i] == 0 else 0
                                response_delay[i] = ai_start - u_end if response_delay[i] == 0 else response_delay[i]
                            else:
                                success_response_to_interruption += 1 if response_delay_to_interruption[i] == 0 else 0
                                if response_delay_to_interruption[i] == 0:
                                    self.count_user_interruptions_response_success(cur_interruption_type)
                                response_delay_to_interruption[i] = ai_start - u_end if response_delay_to_interruption[i] == 0 else response_delay_to_interruption[i]
                                current_user_interrupt = False
                            continue
                            # response[i] = True
                    
            for a_i, ai_end in enumerate(ai_ends):
                for i, u_start in enumerate(user_starts):
                    u_end = user_ends[i]
                    if i < len(user_starts) - 1:
                        next_u_start = user_starts[i+1]
                        if u_end < ai_end < next_u_start and next_u_start - ai_end >= 40000 and ai_end - ai_starts[a_i] < 88000: # 5.5 seconds 
                            count_noise_interrupt += 1
                            continue
                            
                    # if u_start < ai_end < u_end:
                    #     success_interrupt += 1 if interrupt_delay[i - 1] == 0 else 0
                    #     interrupt_delay[i - 1] = ai_end - u_start if interrupt_delay[i - 1] == 0 else interrupt_delay[i - 1]
                    
                    if u_start < ai_end < u_end:
                        if u_start < ai_starts[a_i] < u_end - 8000: # Means that it was previously counted as success_interruption
                            continue
                        else:
                            success_interrupt += 1
                            if interrupt_delay[i - 1] == 0:
                                interrupt_delay[i - 1] = ai_end - u_start
                                self.count_user_interruptions_success(ground_truth.get(i+1).get("Type"))
                            elif type(interrupt_delay[i - 1]) == int:
                                current_delay = [interrupt_delay[i - 1], ai_end - u_start]
                                interrupt_delay[i - 1] = current_delay
                            elif type(interrupt_delay[i - 1]) == list:
                                current_delay = interrupt_delay[i - 1]
                                current_delay.append(ai_end - u_start)
                                interrupt_delay[i - 1] = current_delay
                            
                    
            
            try:
                if self.strict:        
                    assert count_wrong_interrupt_ai == count_greater_than_zero(lead_time_interrupt), "Wrong Interrupts do not match {} vs {}".format(count_wrong_interrupt_ai, count_greater_than_zero(lead_time_interrupt))
                    
                    assert success_interrupt == count_greater_than_zero(interrupt_delay), "Success Interrupts do not match {} vs {}".format(success_interrupt, count_greater_than_zero(interrupt_delay))
                    
                    assert success_response + success_response_to_interruption == count_greater_than_zero(lead_times) + count_greater_than_zero(lead_time_to_interruption) + count_greater_than_zero(response_delay) + count_greater_than_zero(response_delay_to_interruption), "Success Responses do not match {} vs {}".format(success_response, count_greater_than_zero(lead_times) + count_greater_than_zero(response_delay))
            except AssertionError as e:
                if self.debug:
                    # ipdb.set_trace()
                    print("\nConversation ID: ", conv_id)
                    print(user_timestamp)
                    print(ai_timestamp)
                    print("Total Conversation Rounds: ", total_conversation_rounds)
                    print("Success Response: ", success_response)
                    print("Number Interrupt: ", total_real_interruptions)
                    print("Success Interrupt: ", success_interrupt)
                    print("Wrong Interrupt: ", count_wrong_interrupt_ai)
                    print("Noise Interrupt: ", count_noise_interrupt)
                    print("Response Delay: ", response_delay)
                    print("Interrupt Delay: ", interrupt_delay)
                    print("Lead Times: ", lead_times)
                    print("Lead Times Interrupt: ", lead_time_interrupt)
            self.VAD_marks[conv_id] = {"Number Round": total_conversation_rounds, "Success Response": success_response, "Success Response to Interruptions": success_response_to_interruption, "Number Interrupt": total_real_interruptions, "Number Gaps": total_interruptions, "Success Interrupt": success_interrupt, "Wrong Interrupt": count_wrong_interrupt_ai, "Noise Interrupt": count_noise_interrupt, "Response Delay": response_delay, "Response Delay to Interruption": response_delay_to_interruption, "Interrupt Delay": interrupt_delay, "Lead Times": lead_times, "Lead Times to Interruption": lead_time_to_interruption, "Lead Times Interrupt": lead_time_interrupt, "VAD_Details": ai_timestamp}
            
                # print("Response: ", response)
                
    def analyze_VAD_interruption_new2(self):
        """
        Improved version of VAD interruption analysis with enhanced timing logic.
        
        This is an enhanced version of the interruption analysis that provides more accurate
        timing calculations and better handling of edge cases. The analysis is performed
        on a per-user-utterance basis rather than per-AI-utterance for better accuracy.
        
        Key improvements over analyze_VAD_interruption_new:
        - Better handling of multiple AI responses to single user utterances
        - More accurate categorization of response timing (lead times vs delays)
        - Improved detection of noise interruptions and third-party interference
        - Enhanced validation with assertion checks for debugging
        
        The method analyzes:
        - Successful AI responses and their timing
        - AI interruptions during user speech
        - Response delays after user finishes speaking
        - Lead times when AI responds before user finishes
        - Noise interruptions and third-party interference
        
        Updates:
            self.VAD_marks: Comprehensive analysis results including all timing metrics
            and success rates categorized by interruption type.
        """
        # Rewritten. Using User's start/end as the basis.
        # For the AI part, if there is a preamble, discard the first one (moshi).
        # Recurrently go through the AI's start/end:
        #   If the start falls between the user's start-end, it might be a wrong interruption.
        #       If the gap between start to user's end is less than 1 seconds, it is a correct response, calculate the lead-time.
        #       If the gap between start to user's end is more than 1 seconds, it is a wrong interruption.
        #   If the start falls between the user's end-start, it is a correct response, calculate the delay.
        #   If the end falls between the user's start-end, it is a correct interruption, calculate the delay.
        #   If the end falls between the user's end-start, then determine the distance between the end and the user's start, 
        #      If it exceeds 2.5s, it indicates an interruption by a third party (currently only the last segment is implemented as noise data, additional noise interruptions need to be generated).
        self.get_user_inquiry()
        for conv_id, user_timestamp in self.user_timestamps.items():
            ground_truth = self.ground_truth_dict.get(conv_id)
            ai_timestamp = self.ai_timestamps.get(conv_id)
            ai_starts = [a['start'] for a in ai_timestamp]
            ai_ends = [a['end'] for a in ai_timestamp]
            user_starts = [u['start'] for u in user_timestamp]
            user_ends = [u['end'] for u in user_timestamp]
            
            total_conversation_rounds = len(user_timestamp)
            total_interruptions = total_conversation_rounds - 1
            total_real_interruptions = 0
            
            count_wrong_interrupt_ai = 0
            lead_time_interrupt = [0 for _ in range(total_conversation_rounds)] # lead for wrong interrupt
            
            count_noise_interrupt = 0
            
            success_response = 0
            success_response_to_interruption = 0
            response_delay = [0 for _ in range(total_conversation_rounds)] # delay for successful response
            response_delay_to_interruption = [0 for _ in range(total_conversation_rounds)] # delay for successful response to interruption
            lead_times = [0 for _ in range(total_conversation_rounds)]
            lead_time_to_interruption = [0 for _ in range(total_conversation_rounds)]
            
            success_interrupt = 0
            interrupt_delay = [0 for _ in range(total_conversation_rounds)]  # delay for successful interrupt
            
            
            
            if self.model_name == "Moshi":
                # Discard the first AI start/end if there is a preamble
                ai_starts = ai_starts[1:]
                ai_ends = ai_ends[1:]
                

            # We first check AI starts
            
            for i, u_start in enumerate(user_starts):
                current_user_interrupt = False

                for a_i, ai_start in enumerate(ai_starts):
                    u_end = user_ends[i]
                    if ai_start < u_start < ai_ends[a_i]: # (ai_start > u_start and ai_ends[a_i] < u_end)
                        total_real_interruptions += 1
                        cur_interruption_type = self.count_user_interruptions(ground_truth.get(i + 1).get("Type"))
                        current_user_interrupt = True # 1158
                    if u_start < ai_start < u_end - 8000:
                        count_wrong_interrupt_ai += 1
                        if lead_time_interrupt[i] == 0:
                            lead_time_interrupt[i] = u_end - ai_start
                        elif type(lead_time_interrupt[i]) == int:
                            current_lead = [lead_time_interrupt[i], u_end - ai_start]
                            lead_time_interrupt[i] = current_lead
                        elif type(lead_time_interrupt[i]) == list:
                            current_lead = lead_time_interrupt[i]
                            current_lead.append(u_end - ai_start)
                            lead_time_interrupt[i] = current_lead
                            
                        if ai_ends[a_i] < u_end:
                            # total_real_interruptions += 1
                            # cur_interruption_type = self.count_user_interruptions(ground_truth.get(i + 1).get("Type"))
                            # success_interrupt += 1
                            # current_user_interrupt = True # 1158
                            try: 
                                if interrupt_delay[i] == 0:
                                    total_real_interruptions += 1
                                    cur_interruption_type = self.count_user_interruptions(ground_truth.get(i + 1).get("Type"))
                                    success_interrupt += 1
                                    current_user_interrupt = True
                                    interrupt_delay[i] = ai_ends[a_i] - ai_start
                                    if interrupt_delay[i] == 0:
                                        interrupt_delay[i] = 1
                                    self.count_user_interruptions_success(cur_interruption_type)

                                
                            except IndexError:
                                print("Conversation Error: ", conv_id)
                        continue
                    elif u_start < ai_start < u_end:
                        if not current_user_interrupt:
                            success_response += 1
                            lead_times[i] = u_end - ai_start
                        else: 
                            if lead_time_to_interruption[i] == 0 and response_delay_to_interruption[i] == 0:
                                success_response_to_interruption += 1
                                self.count_user_interruptions_response_success(cur_interruption_type)
                                lead_time_to_interruption[i] = u_end - ai_start
                                if lead_time_to_interruption[i] == 0:
                                    lead_time_to_interruption[i] = 1
                                current_user_interrupt = False
                                if interrupt_delay[i] == 0:
                                    success_interrupt += 1
                                    interrupt_delay[i] = ai_ends[a_i] - u_start
                                    if interrupt_delay[i] == 0:
                                        interrupt_delay[i] = 1
                                    self.count_user_interruptions_success(cur_interruption_type)

                        continue
                        
                    
                    if i < len(user_starts) - 1:
                        next_u_start = user_starts[i+1]
                        if ai_start < u_start:
                            if u_start < ai_ends[a_i] <= u_end and current_user_interrupt:
                                if interrupt_delay[i] == 0:
                                    success_interrupt += 1
                                    interrupt_delay[i] = ai_ends[a_i] - u_start
                                    if interrupt_delay[i] == 0:
                                        interrupt_delay[i] = 1
                                    self.count_user_interruptions_success(cur_interruption_type)

                            continue
                        if u_end < ai_ends[a_i] < next_u_start and ai_start < u_start:
                            if current_user_interrupt:
                                if interrupt_delay[i] == 0:
                                    success_interrupt += 1
                                    self.count_user_interruptions_success(cur_interruption_type)
                                    interrupt_delay[i] = ai_ends[a_i] - u_start
                                    if interrupt_delay[i] == 0:
                                        interrupt_delay[i] = 1

                                # current_user_interrupt = False
                                    continue
                        if u_end < ai_start < next_u_start:
                            if not current_user_interrupt:
                                success_response += 1 if response_delay[i] == 0 else 0
                                response_delay[i] = ai_start - u_end if response_delay[i] == 0 else response_delay[i]
                            else:
                                if response_delay_to_interruption[i] == 0 and lead_time_to_interruption[i] == 0:
                                    success_response_to_interruption += 1 
                                    self.count_user_interruptions_response_success(cur_interruption_type)
                                    response_delay_to_interruption[i] = ai_start - u_end
                                    if response_delay_to_interruption[i] == 0:
                                        response_delay_to_interruption[i] = 1
                                    current_user_interrupt = False
                                    if interrupt_delay[i] == 0:
                                        success_interrupt += 1
                                        interrupt_delay[i] = ai_ends[a_i] - u_start
                                        if interrupt_delay[i] == 0:
                                            interrupt_delay[i] = 1
                                        self.count_user_interruptions_success(cur_interruption_type)

                            continue
                            # response[i] = True
                    else:
                        if ai_start < u_start:
                            if u_start < ai_ends[a_i] and current_user_interrupt:
                                if interrupt_delay[i] == 0:
                                    success_interrupt += 1
                                    interrupt_delay[i] = ai_ends[a_i] - u_start
                                    if interrupt_delay[i] == 0:
                                        interrupt_delay[i] = 1
                                    self.count_user_interruptions_success(cur_interruption_type)

                            continue
                        if u_end < ai_start:
                            if not current_user_interrupt:
                                success_response += 1 if response_delay[i] == 0 else 0
                                response_delay[i] = ai_start - u_end if response_delay[i] == 0 else response_delay[i]
                            else:
                                if response_delay_to_interruption[i] == 0 and lead_time_to_interruption[i] == 0:
                                    success_response_to_interruption += 1 
                                    self.count_user_interruptions_response_success(cur_interruption_type)
                                    response_delay_to_interruption[i] = ai_start - u_end
                                    if response_delay_to_interruption[i] == 0:
                                        response_delay_to_interruption[i] = 1
                                    current_user_interrupt = False
                                    if interrupt_delay[i] == 0:
                                        success_interrupt += 1
                                        interrupt_delay[i] = ai_ends[a_i] - u_start
                                        if interrupt_delay[i] == 0:
                                            interrupt_delay[i] = 1
                                        self.count_user_interruptions_success(cur_interruption_type)

                            continue
                    
            for a_i, ai_end in enumerate(ai_ends):
                for i, u_start in enumerate(user_starts):
                    u_end = user_ends[i]
                    if i < len(user_starts) - 1:
                        next_u_start = user_starts[i+1]
                        if u_end < ai_end < next_u_start:
                            if next_u_start - ai_end >= 40000 and ai_end - ai_starts[a_i] < 88000: # 5.5 seconds 
                                count_noise_interrupt += 1
                                continue
                            
                    
            
            try:
                if self.strict:        
                    assert count_wrong_interrupt_ai == count_greater_than_zero(lead_time_interrupt), "Wrong Interrupts do not match {} vs {}".format(count_wrong_interrupt_ai, count_greater_than_zero(lead_time_interrupt))
                    
                    assert success_interrupt == count_greater_than_zero(interrupt_delay), "Success Interrupts do not match {} vs {}".format(success_interrupt, count_greater_than_zero(interrupt_delay))
                    
                    assert success_response + success_response_to_interruption == count_greater_than_zero(lead_times) + count_greater_than_zero(lead_time_to_interruption) + count_greater_than_zero(response_delay) + count_greater_than_zero(response_delay_to_interruption), "Success Responses do not match {} vs {}".format(success_response, count_greater_than_zero(lead_times) + count_greater_than_zero(response_delay))
                    
                    assert success_response_to_interruption <= success_interrupt, "Success Response to Interruption is greater than Success Interruption {} vs {}".format(success_response_to_interruption, success_interrupt)
            except AssertionError as e:
                if self.debug:
                    # ipdb.set_trace()
                    print("\nConversation ID: ", conv_id)
                    print(user_timestamp)
                    print(ai_timestamp)
                    print("Total Conversation Rounds: ", total_conversation_rounds)
                    print("Success Response: ", success_response)
                    print("Number Interrupt: ", total_real_interruptions)
                    print("Success Interrupt: ", success_interrupt)
                    print("success_response_to_interruption: ", success_response_to_interruption)
                    print("Wrong Interrupt: ", count_wrong_interrupt_ai)
                    print("Noise Interrupt: ", count_noise_interrupt)
                    print("Response Delay: ", response_delay)
                    print("response_delay_to_interruption: ", response_delay_to_interruption)
                    print("Interrupt Delay: ", interrupt_delay)
                    print("Lead Times: ", lead_times)
                    print("Lead Times Interrupt: ", lead_time_interrupt)
                    print("lead_time_to_interruption: ", lead_time_to_interruption)
            self.VAD_marks[conv_id] = {"Number Round": total_conversation_rounds, "Success Response": success_response, "Success Response to Interruptions": success_response_to_interruption, "Number Interrupt": total_real_interruptions, "Number Gaps": total_interruptions, "Success Interrupt": success_interrupt, "Wrong Interrupt": count_wrong_interrupt_ai, "Noise Interrupt": count_noise_interrupt, "Response Delay": response_delay, "Response Delay to Interruption": response_delay_to_interruption, "Interrupt Delay": interrupt_delay, "Lead Times": lead_times, "Lead Times to Interruption": lead_time_to_interruption, "Lead Times Interrupt": lead_time_interrupt, "VAD_Details": ai_timestamp}
            
                # print("Response: ", response)
    
    def analyze_VAD_marks(self, output_path: str = None, type: str = "oracle"):
        """
        Aggregate and analyze VAD timing marks to compute final metrics and generate visualizations.
        
        Args:
            output_path (str, optional): Path where analysis results and metrics will be saved.
            type (str, optional): Analysis type - "oracle" for basic metrics or "interruption" 
                for detailed interruption analysis. Defaults to "oracle".
        
        This method processes the VAD analysis results to compute aggregate metrics:
        
        For "oracle" type:
        - Success Response Rate (SRR): Percentage of successful AI responses
        - Wrong Interrupt Rate (IER): Percentage of inappropriate AI interruptions  
        - First Turn Delay (FTD): Average response delay in milliseconds
        - Lead Time (LT): Average lead time for AI responses
        - Wrong Lead Time Interrupt (WLTI): Average lead time for wrong interruptions
        
        For "interruption" type (more comprehensive):
        - All oracle metrics plus:
        - Success Interrupt Rate (SIR): Percentage of successful AI interruptions
        - Noise Interrupt Rate (NIR): Percentage of noise-related interruptions
        - Success Response Rate to Interruption (SRIR): Response success after interruptions
        - Interruption Response Delay (IRD): Timing for responses to interruptions
        - Per-category breakdown of interruption types and success rates
        
        Generates pie charts for timing distributions and saves comprehensive metrics
        including Word Error Rate (WER) from speech recognition analysis.
        
        Updates:
            self.metrics: Dictionary containing all computed metrics
            Creates visualization files and metrics score file at output_path
        """
        # {'Number Round': 4, 'Success Response': 4, 'Number Interrupt': 3, 'Success Interrupt': 0, 'Wrong Interrupt': 0, 'Noise Interrupt': 0, 'Response Delay': [0, 0, 0, 0], 'Interrupt Delay': [0, 0, 0], 'Lead Times': [1472, 8128, 4544, 3008], 'Lead Times Interrupt': [0, 0, 0, 0]}
        os.makedirs(os.path.dirname(os.path.dirname(output_path)), exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if type == 'oracle':
            num_round = 0
            success_response = 0
            wrong_interrupt = 0
            response_delays = list()
            response_delays_to_interruption = list()
            lead_times_to_interruption = list()
            lead_times_interrupt = list()
            lead_times = list()
            
            for _, data in self.VAD_marks.items():
                if data['Number Round'] > 0:
                    num_round += data['Number Round']
                    
                    success_response += data['Success Response']
                    success_response += data['Success Response to Interruptions']
                    
                    wrong_interrupt += data['Wrong Interrupt']
                    
                    response_delays.append(flatten_list(data['Response Delay']))
                    response_delays_to_interruption.append(flatten_list(data['Response Delay to Interruption']))
                    
                    lead_times.append(flatten_list(data['Lead Times']))
                    lead_times_interrupt.append(flatten_list(data['Lead Times Interrupt']))
                    lead_times_to_interruption.append(flatten_list(data['Lead Times to Interruption']))
                    
            self.metrics["Success Respond Rate"] = success_response / num_round
            self.metrics["Wrong Interrupt Rate"] = wrong_interrupt / num_round
            response_delays = flatten_list(response_delays)
            response_delays.append(flatten_list(response_delays_to_interruption))
            response_delays = [int(delay / 16) for delay in flatten_list(response_delays) if delay > 0] # Convert to ms
            lead_times = [int(lead / 16) for lead in flatten_list(lead_times) if lead > 0] + [int(lead / 16) for lead in flatten_list(lead_times_to_interruption) if lead > 0] # Convert to ms
            lead_times_interrupt = [int(lead / 16) for lead in flatten_list(lead_times_interrupt) if lead > 0] # Convert to ms
            self.metrics["Response Delays"] = time_average(response_delays)
            self.metrics["Lead Times"] = time_average(lead_times)
            self.metrics["Wrong Lead Times Interrupt"] = time_average(lead_times_interrupt)
            draw_pie_chart_delay(response_delays, os.path.join(os.path.dirname(output_path), os.path.basename(output_path).split(".")[0]) + " Response Delays", os.path.join(os.path.dirname(output_path), "response_delays.png"))
            draw_pie_chart_lead(lead_times, os.path.join(os.path.dirname(output_path), os.path.basename(output_path).split(".")[0]) + " Lead Times", os.path.join(os.path.dirname(output_path),"lead_times.png"))
            with open(output_path, 'w') as f:
                score = dict() # Keep only 2 decimal points
                score["SRR(%)"] = format(self.metrics["Success Respond Rate"] * 100, '.2f')
                score["IER(%)"] = format(self.metrics["Wrong Interrupt Rate"] * 100, '.2f')
                score["FTD(ms)"] = format(self.metrics["Response Delays"], '.2f')
                score["LT(ms)"] = format(self.metrics["Lead Times"], '.2f')
                score["WLTI(ms)"] = format(self.metrics["Wrong Lead Times Interrupt"], '.2f')
                with open(self.wer, 'r') as wer:
                    wer = wer.readlines()
                    wer_overall = wer[-6]
                    score["WER(%)"] = "".join(wer_overall.split()[2])
                f.write(str(score))
            if self.debug:
                print("Success Response Rate: ", self.metrics["Success Respond Rate"])
                print("Wrong Interrupt Rate: ", self.metrics["Wrong Interrupt Rate"])
                print("Response Delays: {} ms".format(self.metrics["Response Delays"]))
                print("Lead Times: {} ms".format(self.metrics["Lead Times"]))
            
        elif type == 'interruption':
            num_round = 0
            num_interrupt = 0
            num_gap = 0
            success_response = 0
            success_response_to_interrupion = 0
            success_interrupt = 0
            wrong_interrupt = 0
            noise_interrupt = 0
            interruption_delays = list()
            response_delays_to_interruption = list()
            response_delays = list()
            lead_times_to_interruption = list()
            lead_times_interrupt = list()
            lead_times = list()
            
            for _, data in self.VAD_marks.items():
                if data['Number Round'] > 0:
                    num_round += data['Number Round']
                    num_interrupt += data['Number Interrupt']
                    num_gap += data['Number Gaps']
                    
                    success_response += data['Success Response']
                    success_response_to_interrupion += data['Success Response to Interruptions']
                    
                    success_interrupt += data['Success Interrupt']
                    wrong_interrupt += data['Wrong Interrupt']
                    noise_interrupt += data['Noise Interrupt']
                    
                    interruption_delays.append(flatten_list(data['Interrupt Delay']))
                    response_delays.append(flatten_list(data['Response Delay']))
                    response_delays_to_interruption.append(flatten_list(data['Response Delay to Interruption']))
                    response_delays.append(flatten_list(data['Response Delay to Interruption']))
                    
                    lead_times.append(flatten_list(data['Lead Times']))
                    lead_times_interrupt.append(flatten_list(data['Lead Times Interrupt']))
                    lead_times_to_interruption.append(flatten_list(data['Lead Times to Interruption']))
                    
            try: 
                self.metrics["Success Respond Rate"] = success_response / num_round 
            except ZeroDivisionError: 
                self.metrics["Success Respond Rate"] = 0
            try:
                self.metrics["Success Respond Rate to Interruption"] = success_response_to_interrupion / success_interrupt
            except ZeroDivisionError:
                self.metrics["Success Respond Rate to Interruption"] = 0
            try:
                self.metrics["Wrong Interrupt Rate"] = wrong_interrupt / num_round
            except ZeroDivisionError:
                self.metrics["Wrong Interrupt Rate"] = 0
            try:
                self.metrics["Success Interrupt Rate"] = success_interrupt / num_interrupt
            except ZeroDivisionError:
                self.metrics["Success Interrupt Rate"] = 0
            try:
                self.metrics["Noise Interrupt Rate"] = noise_interrupt / num_gap
            except ZeroDivisionError:
                self.metrics["Noise Interrupt Rate"] = 0
            response_delays = [int(delay / 16) for delay in flatten_list(response_delays) if delay > 0] # Convert to ms
            response_delays_to_interruption = [int(delay / 16) for delay in flatten_list(response_delays_to_interruption) if delay > 0] # Convert to ms
            interruption_delays = [int(delay / 16) for delay in flatten_list(interruption_delays) if delay > 0] # Convert to ms
            
            lead_times = [int(lead / 16) for lead in flatten_list(lead_times) if lead > 0] + [int(lead / 16) for lead in flatten_list(lead_times_to_interruption) if lead > 0] # Convert to ms
            lead_times_interrupt = [int(lead / 16) for lead in flatten_list(lead_times_interrupt) if lead > 0] # Convert to ms
            lead_times_to_interruption = [int(lead / 16) for lead in flatten_list(lead_times_to_interruption) if lead > 0] # Convert to ms
            
            self.metrics["Response Delays"] = time_average(response_delays)
            self.metrics["Response Delays to Interruption"] = time_average(response_delays_to_interruption)
            self.metrics["Lead Times"] = time_average(lead_times)
            self.metrics["Wrong Lead Times Interrupt"] = time_average(lead_times_interrupt)
            self.metrics["Lead Times to Interruption"] = time_average(lead_times_to_interruption)
            self.metrics["Interruption Delays"] = time_average(interruption_delays)
            if len(response_delays) > 0:
                draw_pie_chart_delay(response_delays, os.path.join(os.path.dirname(output_path), os.path.basename(output_path).split(".")[0]) + " Response Delays", os.path.join(os.path.dirname(output_path), "response_delays.png"))
            if len(response_delays_to_interruption) > 0:
                draw_pie_chart_delay(response_delays_to_interruption, os.path.join(os.path.dirname(output_path), os.path.basename(output_path).split(".")[0]) + " Response Delays to Interruption", os.path.join(os.path.dirname(output_path), "response_delays_to_interruption.png"))
            if len(interruption_delays) > 0:
                draw_pie_chart_delay(interruption_delays, os.path.join(os.path.dirname(output_path), os.path.basename(output_path).split(".")[0]) + " Interruption Success Delays", os.path.join(os.path.dirname(output_path), "interruption_success_delays.png"))
            if len(lead_times) > 0:
                draw_pie_chart_lead(lead_times, os.path.join(os.path.dirname(output_path), os.path.basename(output_path).split(".")[0]) + " Lead Times", os.path.join(os.path.dirname(output_path),"lead_times.png"))
            if len(lead_times_to_interruption) > 0: 
                draw_pie_chart_delay(lead_times_to_interruption, os.path.join(os.path.dirname(output_path), os.path.basename(output_path).split(".")[0]) + " Lead Times to Interruption", os.path.join(os.path.dirname(output_path), "lead_times_to_interruption.png"))

            with open(output_path, 'w') as f:
                score = dict() # Keep only 2 decimal points
                
                score["SRR(%)"] = format(self.metrics["Success Respond Rate"] * 100, '.2f')
                score["SIR(%)"] = format(self.metrics["Success Interrupt Rate"] * 100, '.2f')
                score["EIR(%)"] = format(self.metrics["Wrong Interrupt Rate"] * 100, '.2f')
                score["NIR(%)"] = format(self.metrics["Noise Interrupt Rate"] * 100, '.2f')
                score["SRIR(%)"] = format(self.metrics["Success Respond Rate to Interruption"] * 100, '.2f')
                
                score["FSED(ms)"] = format(self.metrics["Response Delays"], '.2f')
                score["ERT(ms)"] = format(self.metrics["Lead Times"], '.2f')
                score["EIT(ms)"] = format(self.metrics["Wrong Lead Times Interrupt"], '.2f')
                score["IRD(ms)"] = format(self.metrics["Interruption Delays"], '.2f')
                for key, value in self.interruption_cats.items():
                    score[key + "-all"] = value
                    score[key + "-int-suc"] = self.interruption_success_cats.get(key)
                    score[key + "-res-suc"] = self.interruption_res_suc_cats.get(key)
                
                with open(self.wer, 'r') as wer:
                    wer = wer.readlines()
                    wer_overall = wer[-6]
                    score["WER(%)"] = "".join(wer_overall.split()[2])
                f.write(str(score))

    def analyze_data(self):
        """
        Parse and process input conversation data from files.
        
        This method processes the main input data file to extract:
        - Conversation IDs and response tokens
        - AI and user timestamps for timing analysis
        - Model-specific token processing (especially for Moshi)
        
        For Moshi model:
        - Processes response tokens and maps them to text segments
        - Handles token-to-text conversion using timestamp-based indexing
        - Extracts response text segments for each conversation round
        
        For other models:
        - Extracts AI timestamps without token processing
        - Prepares data structure for analysis without text segmentation
        
        The input data format expects pipe-separated values containing:
        conversation_id || response_tokens || response_text || user_timestamps || ai_timestamps
        
        Updates:
            self.conv_id: List of conversation identifiers
            self.response_text_segments: Dictionary mapping conversation IDs to response text
            self.user_timestamps: Dictionary mapping conversation IDs to user timing data
            self.ai_timestamps: Dictionary mapping conversation IDs to AI timing data
        """
        for conversation in self.all_data:
            conversation = conversation.strip().split("||")
            self.conv_id.append(conversation[0].strip())
            # self.response_token_id.append(ast.literal_eval(conversation[1].strip()))  # Later we will get the response text from the token instead of response text
            if self.model_name == "Moshi":
                response_tokens = ast.literal_eval(conversation[1].strip())
                ai_timestamp = ast.literal_eval(conversation[4].strip())
                # Only for Moshi?
                idx = 0
                response_token_text = []
                response_word_text = ast.literal_eval(conversation[2].strip())
                for time_dict in ai_timestamp:
                    start = time_dict.get("start") // 1280 - 1
                    end = time_dict.get("end") // 1280 + 1
                    start_idx = idx
                    for token in response_tokens[start:end]:
                        if token not in [0, 3]:
                            idx += 1
                    response_token_text.append("".join(response_word_text[start_idx: idx]))
            else:
                ai_timestamp = ast.literal_eval(conversation[4].strip())
                response_token_text = []
                                
            # self.response_text.append("".join(ast.literal_eval(conversation[2].strip()))) # Used to compare with ASR results --> WER
            self.response_text_segments[conversation[0].strip()] = response_token_text
            # self.user_timestamps.append(ast.literal_eval(conversation[3].strip()))
            # self.ai_timestamps.append(ai_timestamp)
            self.user_timestamps[conversation[0].strip()] = ast.literal_eval(conversation[3].strip())
            self.ai_timestamps[conversation[0].strip()] = ai_timestamp
            

    # Now, we are going to use WhisperX to decode the responding text to check
    #     1. WER (compare with the response text)
    #     2. Store the text for further analysis
    
    def initial_whisperx_module(self):
        """
        Initialize the WhisperX ASR (Automatic Speech Recognition) model.
        
        Loads the WhisperX large-v3-turbo model with specified device and compute type
        for speech recognition of AI responses. This model will be used to transcribe
        AI-generated audio to text for Word Error Rate (WER) calculation.
        
        Updates:
            self.asr_model: Loaded WhisperX model instance ready for transcription
        """
        self.asr_model = whisperx.load_model("large-v3-turbo", self.asr_device, compute_type=self.asr_compute_type)

    def initial_asr_output(self):
        """
        Initialize file paths for ASR output and WER calculation.
        
        Creates directory structure and sets up file paths for:
        - Hypothesis file (ASR transcribed text)
        - Reference file (ground truth text)  
        - WER results file
        
        Updates:
            self.hyp: Path to hypothesis file for ASR output
            self.ref: Path to reference file for ground truth text
            self.wer: Path to WER calculation results file
        """
        ASR_folder = os.path.join(os.path.dirname(self.input_data), os.path.basename(self.input_data).split(".")[0] + "_asr")
        os.makedirs(ASR_folder, exist_ok=True)

        self.hyp = os.path.join(ASR_folder, "hyp")
        self.ref = os.path.join(ASR_folder, "ref")
        self.wer = os.path.join(ASR_folder, "wer")
    
    def parse_data_audio_asr(self):
        """
        Process audio data through ASR for quality evaluation and WER calculation.
        
        This method handles the complete pipeline for audio processing and speech recognition:
        
        For Moshi model:
        1. Initializes WhisperX ASR model and output file structure
        2. Processes each conversation audio file through ASR
        3. Maps AI responses to conversation rounds based on timing
        4. Generates hypothesis and reference files for WER calculation
        5. Associates transcribed AI text with conversation context
        
        For other models:
        - Processes audio files without generating WER reference files
        - Still performs ASR transcription and conversation mapping
        
        The method creates a quality check dictionary that combines:
        - Original ground truth conversation data
        - ASR-transcribed AI responses
        - Timing information for response analysis
        
        Updates:
            self.quality_check_dict: Dictionary containing complete conversation data
            with both user inputs and ASR-transcribed AI responses mapped to rounds.
            For Moshi: Also creates hypothesis and reference files for WER calculation.
        """
        if self.model_name == "Moshi":
            # first, initial the model for ASR
            self.initial_whisperx_module()
            self.initial_asr_output()
            # Then, parse the data folder containing all output audios
            speech_folder = os.path.join(os.path.dirname(self.input_data), os.path.basename(self.input_data).split(".")[0])
            hyp = open(self.hyp, 'w')
            ref = open(self.ref, 'w')
            
            for speech_name in os.listdir(speech_folder):
                # {"User": value.get("user"), "Type": next_interruption, "Timestamps": user_timestamps[int(key.split("-")[1]) - 1]}
                self.quality_check_dict[speech_name] = {key: value for key, value in self.ground_truth_dict.get(speech_name).items()}
                User_start_times = list()
                for key in self.quality_check_dict[speech_name].keys():
                    self.quality_check_dict[speech_name][key]["AI"] = ""
                    self.quality_check_dict[speech_name][key]["AI_Timestamps"] = []
                    User_start_times.append(self.quality_check_dict[speech_name][key].get("Timestamps").get("start"))
                speech_path = os.path.join(speech_folder, speech_name)
                result, AI_start_times, AI_time_stamps = self.recognize_response_speech_using_whisperx(speech_name, speech_path)
                AI_speak_idx = find_ai_speaks_idx(AI_start_times, User_start_times)
                AI_responses = list()
                # write ref, hyp into two files
                for idx, each in enumerate(result):
                    hyp_format = "{}-{} {}\n".format(speech_name.split(".")[0], idx, each)
                    ref_format = "{}-{} {}\n".format(speech_name.split(".")[0], idx, self.response_text_segments[speech_name][idx+1])
                    hyp.write(hyp_format)
                    ref.write(ref_format)
                    AI_responses.append(each)
                # for idx in AI_speak_idx:
                #     self.quality_check_dict[speech_name][idx+1]["AI"] = self.quality_check_dict[speech_name][idx+1]["AI"] + AI_responses[idx]
                #     self.quality_check_dict[speech_name][idx+1]["AI_Timestamps"] = self.quality_check_dict[speech_name][idx+1]["AI_Timestamps"].append(AI_time_stamps[idx])
                for idx, each in enumerate(AI_responses):
                    self.quality_check_dict[speech_name][AI_speak_idx[idx]+1]["AI"] += each
                    self.quality_check_dict[speech_name][AI_speak_idx[idx]+1]["AI_Timestamps"].append(AI_time_stamps[idx])
            hyp.close()
            ref.close()
        else:
            # first, initial the model for ASR
            self.initial_whisperx_module()
            self.initial_asr_output()
            # Then, parse the data folder containing all output audios
            speech_folder = os.path.join(os.path.dirname(self.input_data), os.path.basename(self.input_data).split(".")[0])
            
            for speech_name in os.listdir(speech_folder):
                # {"User": value.get("user"), "Type": next_interruption, "Timestamps": user_timestamps[int(key.split("-")[1]) - 1]}
                self.quality_check_dict[speech_name] = {key: value for key, value in self.ground_truth_dict.get(speech_name).items()}
                User_start_times = list()
                for key in self.quality_check_dict[speech_name].keys():
                    self.quality_check_dict[speech_name][key]["AI"] = ""
                    self.quality_check_dict[speech_name][key]["AI_Timestamps"] = []
                    User_start_times.append(self.quality_check_dict[speech_name][key].get("Timestamps").get("start"))
                speech_path = os.path.join(speech_folder, speech_name)
                result, AI_start_times, AI_time_stamps = self.recognize_response_speech_using_whisperx(speech_name, speech_path)
                AI_speak_idx = find_ai_speaks_idx(AI_start_times, User_start_times)
                AI_responses = list()
                # write ref, hyp into two files
                for idx, each in enumerate(result):
                    AI_responses.append(each)
                for idx, each in enumerate(AI_responses):
                    self.quality_check_dict[speech_name][AI_speak_idx[idx]+1]["AI"] += each
                    self.quality_check_dict[speech_name][AI_speak_idx[idx]+1]["AI_Timestamps"].append(AI_time_stamps[idx])

        
        
    def calculate_wer(self):
        """
        Calculate Word Error Rate (WER) between reference and hypothesis text.
        
        Runs the WER calculation script to compare ASR-transcribed AI responses (hypothesis)
        against the ground truth response text (reference). This provides a measure of
        speech recognition accuracy for the AI's audio output.
        
        Only supported for Moshi model which generates both audio and reference text.
        Uses character-level WER calculation with verbose output.
        
        Raises:
            NotImplementedError: If called on models other than Moshi that don't have
            reference text for comparison.
            
        Updates:
            Creates WER results file at self.wer path containing detailed error analysis.
        """
        if self.model_name == "Moshi":
            command = [
                "python",
                "./compute-wer.py",
                "--char=1",
                "--v=1",
                self.ref,
                self.hyp,
            ]
            with open(self.wer, "w") as outfile:
                subprocess.run(command, stdout=outfile, stderr=subprocess.PIPE, text=True)
        else:
            raise NotImplementedError("Calculating WER only supports models that output response text.")
            
 
    def recognize_response_speech_using_whisperx(self, audio_name:str, input_audio: str):
        """
        Transcribe AI response audio using WhisperX ASR for specific conversation segments.
        
        Args:
            audio_name (str): Name of the audio file (conversation identifier)
            input_audio (str): Path to the audio file to transcribe
            
        Returns:
            tuple: A tuple containing:
                - results (list): List of transcribed text segments
                - start_times (list): List of start times for each segment
                - VAD_data (list): Voice activity detection data for the segments
        
        This method:
        1. Loads the audio file and retrieves VAD (Voice Activity Detection) data
        2. For Moshi model, skips the first VAD segment (preamble)
        3. Extracts audio chunks based on VAD timing information
        4. Transcribes each chunk using the WhisperX model
        5. Returns transcribed text aligned with timing data
        
        The transcription is performed with batch processing for efficiency and
        language is set to English for optimal performance.
        """
        audio = whisperx.load_audio(input_audio)
        VAD_data = self.VAD_marks[audio_name].get("VAD_Details")
        audios = list()
        results = list()
        start_times = list()
        if self.model_name == "Moshi":
            VAD_data = VAD_data[1:]
        for data in VAD_data:
            start = data.get("start")
            end = data.get("end")
            start_times.append(start)
            audio_chunk = audio[start:end]
            audios.append(audio_chunk)
        for audio in audios:
            result = self.asr_model.transcribe(audio, batch_size=4, language="en")
            results.append(" ".join(each.get("text") for each in result.get("segments")))
        return results, start_times, VAD_data
        
        
    def write_conversation_rounds(self, output_path: str = None):
        """
        Generate formatted conversation data for LLM (Language Model) analysis.
        
        Args:
            output_path (str, optional): Path where conversation rounds will be saved.
        
        This method processes the quality check dictionary to create a structured format
        suitable for Language Model analysis. It:
        
        1. Combines user inputs and AI responses into conversation context
        2. Builds cumulative conversation history for each round
        3. Associates interruption types with each conversation round
        4. Formats data for downstream perplexity analysis with LLM
        
        The output format includes:
        - Conversation ID mapped to round data
        - Each round contains cumulative conversation content and interruption type
        - Progressive context building (each round includes previous conversation)
        
        This data is typically used for:
        - Conversation quality analysis with Language Models
        - Perplexity calculations for response appropriateness
        - Subjective evaluation of conversation flow
        
        Updates:
            Creates a formatted text file at output_path containing conversation data
            structured for LLM processing and analysis.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        output_round_for_llm = dict()
        for key, value in self.quality_check_dict.items():
            # key = conversation_1.wav
            # value = {id: {User: "Any advice on starting a meditation practice?", AI: "Begin with short sxxx as you become more comfortable.", Type: "First Inquiry", Timestamps: {start: 0, end: 1000}, AI_Timestamps: [{start: 0, end: 1000}]}}
            # f.write("{}.wav: {}\n".format(key, str(value)))
            conv_id = key.split(".")[0]
            context = ""
            new_dict_for_llm = dict()
            for idx, content in value.items():
                # idx = round_id
                # content = {User: "Any advice on starting a meditation practice?", AI: "Begin with short sxxx as you become more comfortable.", Type: "First Inquiry", Timestamps: {start: 0, end: 1000}, AI_Timestamps: [{start: 0, end: 1000}]}
                new_dict_for_llm[idx] = {"Content": context + "User: {}. AI: {}. ".format(content.get("User"), content.get("AI")), "Type": content.get("Type")}
                context = new_dict_for_llm[idx].get("Content")
            output_round_for_llm[conv_id] = new_dict_for_llm
        with open(output_path, 'w') as f:
            f.write(str(output_round_for_llm))
            
    def get_ppl_llama3(self, conversation_round_text: str):
        """
        Aggregate perplexity scores from Llama3 model analysis.
        
        Args:
            conversation_round_text (str): Path to the conversation rounds text file.
        
        This method processes perplexity results generated by Llama3 analysis to:
        1. Read conditional perplexity (CPPL) scores for each conversation round
        2. Calculate average perplexity per conversation
        3. Compute overall average perplexity across all conversations
        4. Generate summary statistics for conversation quality assessment
        
        The perplexity scores provide insights into:
        - How natural/expected the AI responses are given the conversation context
        - Quality of conversation flow and appropriateness
        - Comparative analysis across different interruption types
        
        Input file format expected: .cppl.txt containing structured perplexity data
        Output file format: .cppl.score containing aggregated statistics
        
        Updates:
            Creates a score file containing per-conversation and overall perplexity
            statistics for quantitative conversation quality assessment.
        """
        ppl_result_file = conversation_round_text.replace(".txt", ".cppl.txt")
        output_ppl = conversation_round_text.replace(".txt", ".cppl.score")
        output_ppl_dict = dict()
        avg_ppl_list = list()
        with open(ppl_result_file, 'r') as f:
            ppl_result = f.read()
            ppl_data = ast.literal_eval(ppl_result)
            ppl_value = list()
            ppl_dict = dict()
            for conv_id, data in ppl_data.items():
                for round, ppl in data.items():
                    if ppl.get("CPPL") != "N/A":
                        ppl_value.append(ppl.get("CPPL"))
                        ppl_dict[round] = {"CPPL": ppl.get("CPPL"), "Type": ppl.get("Type")}
                        
                average_ppl_conv = sum(ppl_value) / len(ppl_value)
                avg_ppl_list.append(average_ppl_conv)
                output_ppl_dict[conv_id] = {"PPL": ppl_dict, "Average PPL": average_ppl_conv}
            output_ppl_dict["Average PPL"] = sum(avg_ppl_list) / len(avg_ppl_list)
            with open(output_ppl, 'w') as f:
                f.write(str(output_ppl_dict))
        
                    
def main_moshi():
    """
    Main function for benchmarking Moshi conversational AI model.
    
    Runs complete evaluation pipeline for Moshi including:
    1. Data parsing and timestamp analysis
    2. VAD-based interruption analysis  
    3. Audio processing and ASR transcription
    4. WER calculation for speech quality
    5. Objective metrics computation and visualization
    6. Subjective analysis preparation with conversation round formatting
    7. Perplexity analysis with Llama3
    
    Expects input data path as command line argument.
    Generates comprehensive evaluation results in both objective and subjective metrics folders.
    """
    input_data = sys.argv[1]
    # input_data = "data/Moshi-output/subjective_metrics/cosyvoice2-single-round-combine-hard/conversation_rounds.txt"
    ground_truth = "openai/data_gen/output_all.jsonl"
    benchmarking = Benchmarking(input_data, ground_truth, "interruption")
    benchmarking.analyze_data()
    benchmarking.initial_asr_output()
    benchmarking.analyze_VAD_interruption_new2()

    benchmarking.parse_data_audio_asr()
    benchmarking.calculate_wer()
    output_path = os.path.join(os.path.dirname(input_data), "objective_metrics", os.path.basename(input_data).split(".")[0], "metrics.scores")
    output_text = os.path.join(os.path.dirname(input_data), "subjective_metrics", os.path.basename(input_data).split(".")[0], "conversation_rounds.txt")
    benchmarking.analyze_VAD_marks(output_path=output_path, type="interruption")
    benchmarking.write_conversation_rounds(output_path=output_text)
    benchmarking.get_ppl_llama3(output_text)


def main_freeze_omni():
    """
    Main function for benchmarking Freeze-omni conversational AI model.
    
    Similar to main_moshi but adapted for Freeze-omni model specifics:
    - Uses "Freeze-omni" model name for different processing logic
    - Skips WER calculation (not supported for this model)
    - Runs same interruption analysis and timing evaluation
    - Generates objective and subjective metrics
    
    Expects input data path as command line argument.
    """
    input_data = sys.argv[1]
    # input_data = "data/Freeze-omni-output/subjective_metrics/cosyvoice2-single-round-combine-easy/conversation_rounds.txt"
    ground_truth = "openai/data_gen/output_all.jsonl"
    benchmarking = Benchmarking(input_data, ground_truth, "interruption", model_name="Freeze-omni")
    benchmarking.analyze_data()
    benchmarking.initial_asr_output()

    benchmarking.analyze_VAD_interruption_new2()

    benchmarking.parse_data_audio_asr()
    # benchmarking.calculate_wer()
    output_path = os.path.join(os.path.dirname(input_data), "objective_metrics", os.path.basename(input_data).split(".")[0], "metrics.scores")
    output_text = os.path.join(os.path.dirname(input_data), "subjective_metrics", os.path.basename(input_data).split(".")[0], "conversation_rounds.txt")
    benchmarking.analyze_VAD_marks(output_path=output_path, type="interruption")
    benchmarking.write_conversation_rounds(output_path=output_text)
    benchmarking.get_ppl_llama3(output_text)

def main_vita():
    """
    Main function for benchmarking VITA conversational AI model.
    
    Evaluation pipeline for VITA model with same structure as Freeze-omni:
    - Uses "Freeze-omni" model name for processing (similar architecture)
    - Skips WER calculation (not supported for this model)
    - Complete interruption and timing analysis
    - Generates comprehensive evaluation metrics
    
    Expects input data path as command line argument.
    """
    input_data = sys.argv[1]
    # input_data = "data/VITA-1.5/subjective_metrics/cosyvoice2-single-round-combine-easy/conversation_rounds.txt"
    ground_truth = "openai/data_gen/output_all.jsonl"
    benchmarking = Benchmarking(input_data, ground_truth, "interruption", model_name="Freeze-omni")
    benchmarking.analyze_data()
    benchmarking.initial_asr_output()

    benchmarking.analyze_VAD_interruption_new2()

    benchmarking.parse_data_audio_asr()
    # benchmarking.calculate_wer()
    output_path = os.path.join(os.path.dirname(input_data), "objective_metrics", os.path.basename(input_data).split(".")[0], "metrics.scores")
    output_text = os.path.join(os.path.dirname(input_data), "subjective_metrics", os.path.basename(input_data).split(".")[0], "conversation_rounds.txt")
    benchmarking.analyze_VAD_marks(output_path=output_path, type="interruption")
    benchmarking.write_conversation_rounds(output_path=output_text)
    benchmarking.get_ppl_llama3(output_text)

if __name__ == "__main__":
    main_moshi() # Or other models like main_freeze_omni() or main_vita()
    # main_freeze_omni()
    # main_vita()