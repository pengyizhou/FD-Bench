#!/usr/bin/env python

import argparse, ast
from openai import OpenAI
import time
import json
from tqdm import tqdm
# client = OpenAI()


SYSTEM_PROMPT = {
    "role": "system",
    "content": """You are an evaluator for user–AI interactions. Your task is to:
1. Read the user’s input and the AI’s response. Keep in mind that there might be multi-round User-AI interactions.
2. The evaluation only applies to the final round AI's response.
3. Provide the reasoning for your assessment in numbered points, giving a score from 1 to 10 for each point.
4. Provide a final score from 1 to 10 based on the sub-scores for each point.

Please follow this format exactly:
Reasoning:
	1. Relevance: ...
		Score: X/10
	2. Encouragement: ...
		Score: X/10
	3. Simplicity: ...
		Score: X/10
	4. Flexibility: ...
		Score: X/10
	5. Practicality: ...
		Score: X/10
	6. Creativity: ...
		Score: X/10
...
Overall: X/10

- Always provide Reasoning first, in numbered format, to explain why you are giving the score.
- Then provide the Score, in the form X/10.
- Ensure the reasons logically connect to the score you give.
Use this style and format whenever you assess any user–AI interaction.

There is a suggested scoring framework for each of the six perspectives (Relevance, Encouragement, Simplicity, Flexibility, Practicality, Creativity). For each perspective, there are 10 possible scores, with brief explanations of what each score represents.
Relevance:
	Score 1: Completely off-topic or contradicts the user’s request, showing no alignment with the conversation context.
	Score 2: Barely touches on the topic or question, with minimal alignment to the user’s needs.
	Score 3: Mentions the user’s topic, but remains mostly unrelated or tangential.
	Score 4: Some alignment with the user’s question, but still includes distracting or off-topic elements.
	Score 5: Partially addresses the question, yet important details are missing or unclear.
	Score 6: Mostly aligned with the question but may lack depth or miss certain aspects of the context.
	Score 7: Addresses the main topic directly and aligns well with the user’s request, with minor gaps.
	Score 8: Highly relevant, covering the primary focus of the user’s question without straying off-topic.
	Score 9: Extremely relevant, addressing almost every detail with clarity and focus.
	Score 10: Perfectly on-point, thoroughly covering the user’s request with complete alignment to context.
Encouragement:
	Score 1: Negative or discouraging tone, potentially undermining the user’s confidence or interest.
	Score 2: Slightly discouraging or indifferent, offering little to no positive reinforcement.
	Score 3: Neutral tone, lacking warmth or reassurance for the user.
	Score 4: Shows mild willingness to help but minimal supportive language or motivation.
	Score 5: Moderately positive, offering some acknowledgment of the user’s needs.
	Score 6: Polite and lightly encouraging, though lacking strong enthusiasm or guidance.
	Score 7: Friendly and supportive, conveying clear intent to help and reassure.
	Score 8: Actively encouraging and motivating, showing genuine interest in assisting the user.
	Score 9: Strongly encouraging, with helpful, positive reinforcement that inspires confidence.
	Score 10: Exceptionally uplifting and motivational, making the user feel fully supported and empowered.
Simplicity:
	Score 1: Overly complex, unclear, or confusing; difficult to follow.
	Score 2: Largely convoluted or cluttered with unnecessary details.
	Score 3: Some clarity, but essential instructions or points remain hard to understand.
	Score 4: Understandable in parts but still contains ambiguous or scattered elements.
	Score 5: Somewhat clear, though it may require re-reading or inferring to grasp fully.
	Score 6: Fairly easy to follow, with only a few unclear details.
	Score 7: Mostly straightforward and readable, with minimal confusion.
	Score 8: Clearly presented, logically structured, and easy to process.
	Score 9: Very concise and user-friendly, offering instructions or explanations in an effortless manner.
	Score 10: Exceptionally clear and simple, requiring minimal effort to understand; thoroughly user-centric.
Flexibility:
	Score 1: Entirely rigid, offering no alternatives or adaptability to user preferences.
	Score 2: Very narrow scope, allowing for minimal variation in approach or preference.
	Score 3: Mentions alternatives vaguely but does not provide meaningful options.
	Score 4: Some recognition of different needs, but not integrated well into the answer.
	Score 5: Offers limited adaptability, addressing at least one variation.
	Score 6: Reasonably flexible, suggesting a couple of alternatives without deep detail.
	Score 7: Gives multiple options or strategies, letting the user tailor it to their needs.
	Score 8: Clearly acknowledges diverse preferences and provides structured alternatives.
	Score 9: Highly adaptive, accommodating a wide range of scenarios and user requirements.
	Score 10: Extremely flexible, offering multiple paths or solutions and encouraging further customization by the user.
Practicality:
	Score 1: Entirely impractical or irrelevant for real-world application.
	Score 2: Very difficult to implement, with unclear or unrealistic steps.
	Score 3: Theoretically possible but likely difficult in practice.
	Score 4: Some realistic elements, but lacks feasibility or actionable guidance.
	Score 5: Moderately practical, though it might be missing critical considerations (time, resources).
	Score 6: Mostly usable in a real-world scenario, with a few potential hurdles.
	Score 7: Solidly practical, giving users enough detail to implement relatively easily.
	Score 8: Very practical, well-aligned with typical constraints (time, cost, skill level).
	Score 9: Highly applicable, thoroughly considering ease of use, resource availability, and efficiency.
	Score 10: Exceptionally actionable, presenting a solution or advice that is almost immediately implementable with minimal effort or adjustment.
Creativity:
	Score 1: Completely unoriginal or derivative, adding no new ideas to the conversation.
	Score 2: Mostly uninspired, reusing well-known or stereotypical thoughts.
	Score 3: Slight sparks of originality, but largely formulaic.
	Score 4: Some hint of new angles, yet still heavily reliant on standard solutions.
	Score 5: Moderately creative, suggesting at least one interesting twist or idea.
	Score 6: Displays a few fresh elements, though it may rely partly on common suggestions.
	Score 7: Noticeably inventive, introducing original ideas or combinations.
	Score 8: Impressive creativity, offering unique and thought-provoking suggestions or approaches.
	Score 9: Highly innovative, consistently bringing exciting, unconventional concepts to the table.
	Score 10: Exceptionally original, showcasing a breakthrough or visionary level of creativity that goes far beyond standard expectations.
Here is a few examples.

Example 1:
Input:
[
  {"role": "user", "content": "User: Hey, do you have any easy recipes for a quick dinner? AI: A stir-fry is quick and versatile. Use your favorite vegetables and protein. Season with soy sauce and garlic for flavor?"}
]

Output:

  {"Reasoning":
    "1.	Relevance:
	The response directly addresses the user’s request for a quick dinner recipe and offers a universally simple suggestion. Score: 8/10
    2.	Encouragement:
	The response is concise and assumes the user can handle the recipe. However, it does not explicitly encourage or build confidence. Score: 6/10
    3.	Simplicity:
	The suggestion is simple and easy to follow, perfect for a quick meal. However, more specific steps or ingredient amounts could make it even clearer. Score: 7/10
    4.	Flexibility:
	The suggestion is highly flexible, allowing the user to choose their preferred vegetables and protein. Score: 9/10
    5.	Practicality:
	Stir-frying is practical and fast, but the response lacks additional tips, like suggesting pre-cut vegetables or pre-cooked proteins to save time. Score: 8/10
    6.	Creativity:
	The response is helpful but not particularly creative. Stir-fry is a standard go-to meal for quick dinners. Score: 5/10",
    "Overall": "7.2/10"
}

Example 2:
Input:
[
  {"role": "user", "content": "User: Hey, do you have any easy recipes for a quick dinner? AI: That' a great question. I would recommend starting with a classic recipe like a roasted chicken. It' a great way to learn the basics of cooking and can be applied to so many different dishes. User: Got it. Oh, how about a new show to binge-watch? Anything exciting? AI: Oh that's a good question. One sign is if the leaves start to droop and turn yellow, it's usually because they're not getting enough water. Another sign is if the soil is constantly wet and there's no drainage holes in the pot, that can also lead to root rot."}
]

Output:
{"Reasoning":
    "1.	Relevance:
	The AI’s response is completely irrelevant to the user’s question about a new show to binge-watch. It instead discusses plant care, which is unrelated to both the question and the previous conversation context. Score: 1/10
    2.	Encouragement:
	The response does not acknowledge or build upon the user’s interest in finding an exciting show, nor does it encourage further discussion on the topic. Score: 2/10
    3.	Simplicity:
	While the response itself is simple and understandable, it has no connection to the context or the user’s question, rendering its simplicity unhelpful. Score: 3/10
    4.	Flexibility:
	The response demonstrates no flexibility in addressing the user’s question or context, as it veers off-topic entirely. Score: 1/10
    5.	Practicality:
	The information about plant care might be practical in a different context, but in this case, it is useless and does not help the user find a new show to binge-watch. Score: 1/10
    6.	Creativity:
	There is no creativity in the response. It not only fails to suggest an exciting show but also offers unrelated, out-of-context information. Score: 1/10

    "Overall": "1.5/10"
}
"""
}

def main(args):
    # time.sleep(60)
    # output_file_path = args.input.replace(".txt", ".score.txt")
    output_batch_path = args.input.replace(".txt", ".batch.jsonl")
    with open(args.input, 'r') as f:
        data = f.read()
        data_dict = ast.literal_eval(data)
        request = list()
        for key, value in tqdm(data_dict.items(), desc="Processing", total=len(data_dict)):
            conv_id = key
            # print("start processing {}".format(conv_id))
            score_dict = dict()
            for round, content in value.items():
                type = content.get('Type')
                context = content.get('Content')
                if context.endswith("AI: . "):
                    score_dict[round] = {"Score": "N/A", "Reason": "Failed response", "Content": context, "Type": type}
                else:
                    request.append({"custom_id": "{}-{}".format(conv_id, str(round)), "method":"POST", "url": "/v1/chat/completions", "body": {"model": args.model, "temperature": 0.1, "messages": [SYSTEM_PROMPT, {"role": "user", "content": context}]}})
        with open(output_batch_path, 'w') as f2:
            for item in request:
                json.dump(item, f2)
                f2.write("\n")
            # json.dump(request, f2, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the evaluation for the conversations")
    parser.add_argument("--input", type=str, required=True, help="The input file containing the conversations")
    # parser.add_argument("--output", type=str, help="The output file to save the scores")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-11-20", help="The model to use for the evaluation")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    main(args)
