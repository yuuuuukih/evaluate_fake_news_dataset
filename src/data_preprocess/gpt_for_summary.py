import os
from openai import OpenAI

import sys
import time
import openai
import functools

# Define the retry decorator
def retry_decorator(max_error_count=10, retry_delay=1): # Loop with a maximum of 10 attempts
    def decorator_retry(func):
        functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Initialize an error count
            error_count = 0
            while error_count < max_error_count:
                try:
                    v = func(*args, **kwargs)
                    return v
                except openai.error.Timeout as e:
                    print("Timeout error occurred. Re-running the function.")
                    print(f"Timeout error: {e}")
                    error_count += 1
                except openai.error.APIError as e:
                    print("OPENAI API error occurred. Re-running the function.")
                    print(f"OPENAI API error: {e}")
                    error_count += 1
                except ValueError as e:
                    print("ValueError occurred. Re-running the function.")
                    print(f"ValueError: {e}")
                    error_count += 1
                except AttributeError as e: # For when other functions are called in function calling.
                    print(f"AttributeError occurred: {e}. Re-running the function.")
                    error_count += 1
                except TypeError as e: # For when other functions are called in function calling.
                    print("TypeError occurred. Re-running the function.")
                    print(f"TypeError: {e}")
                    error_count += 1
                except openai.error.InvalidRequestError as e:
                    print("InvalidRequestError occurred. Continuing the program.")
                    print(f"openai.error.InvalidRequestError: {e}")
                    break  # Exit the loop
                except Exception as e:
                    print("Exception error occurred. Re-running the function.")
                    print(f"Exeption: {e}")
                    error_count += 1
                time.sleep(retry_delay)  # If an error occurred, wait before retrying
            if error_count == max_error_count:
                sys.exit("Exceeded the maximum number of retries. Exiting the function.")
                return None
        return wrapper
    return decorator_retry

@retry_decorator(max_error_count=10, retry_delay=1)
def get_summarized_content(input_content: str, words: int = 200, model_name: str = 'gpt-4-1106-preview', temperature: float = 0) -> str:
    # Create prompt.
    prompt_file_path = os.path.join(os.path.dirname(__file__), 'few_shot_prompt.txt')
    with open(prompt_file_path, 'r') as F:
        few_shot_prompt = F.read()

    system_prompt = ''
    user_prompt = f'Summarize the following news article in about {words} words.\n'
    user_prompt += f"{few_shot_prompt}\n"
    user_prompt += (
        "\n"
        "Input:\n"
        f"{input_content}\n"
        "\n"
        "Output:\n"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    # Create client.
    client = OpenAI(
        organization = os.environ['OPENAI_KUNLP'],
        api_key = os.environ['OPENAI_API_KEY_TIMELINE']
    )
    # Get response.
    response = client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        messages=messages
    )

    summarized_content = response.choices[0].message.content
    return summarized_content