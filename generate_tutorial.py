
def generate_makeup_tutorial(facial_features, makeup_style):
    # Generate makeup tutorial based on the provided facial features and style
    print(f"Generating makeup tutorial for {makeup_style} style.")
    print(f"Using facial features: {facial_features}")

    # Here you can implement logic to create a personalized tutorial based on the features
    if makeup_style == "natural":
        tutorial = "1. Apply light foundation.\n2. Light eye makeup.\n3. Subtle lip gloss."
    elif makeup_style == "glam":
        tutorial = "1. Full coverage foundation.\n2. Bold eyeshadow.\n3. Dramatic lipstick."
    else:
        tutorial = "1. Customize your look based on facial features!"

    return tutorial


# # import openai

# # openai.api_key = 'your-openai-api-key'

# # def generate_makeup_tutorial(features):
# #     prompt = f"Based on the following facial features: {features}, provide a step-by-step makeup tutorial. The face has the following features: {features}. Make sure the tutorial is suitable for this face."

# #     response = openai.Completion.create(
# #         engine="text-davinci-003",  # Or the latest model you want to use
# #         prompt=prompt,
# #         max_tokens=200,
# #         n=1,
# #         stop=None,
# #         temperature=0.7
# #     )
    
# #     tutorial = response.choices[0].text.strip()
# #     return tutorial

# import os
# import openai


# # This will print "Hello, World!" to the terminal
# print("Hello, World!")

# openai.api_key = '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'

# def generate_makeup_tutorial(features):
#     prompt = f"Based on the following facial features: {features}, provide a step-by-step makeup tutorial."
#     retries = 3
#     for _ in range(retries):
#         try:
#             response = openai.Completion.create(
#                 engine="text-davinci-003",
#                 prompt=prompt,
#                 max_tokens=200,
#                 n=1,
#                 stop=None,
#                 temperature=0.7
#             )
#             tutorial = response.choices[0].text.strip()
#             return tutorial
#         except openai.error.RateLimitError as e:
#             print(f"Rate limit exceeded. Retrying in 5 seconds...")
#             time.sleep(5)  # Wait before retrying
#         except Exception as e:
#             print(f"An error occurred: {e}")
#             break

#     return "Unable to generate a tutorial due to an error."



# # from openai import OpenAI
# # client = OpenAI()

# # completion = client.chat.completions.create(
# #     model="gpt-4o-mini",
# #     messages=[
# #         {"role": "system", "content": "You are a helpful assistant."},
# #         {
# #             "role": "user",
# #             "content": "Write a haiku about recursion in programming."
# #         }
# #     ]
# # )

# # print(completion.choices[0].message)