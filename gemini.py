from openai import OpenAI
import requests
from bs4 import BeautifulSoup

client = OpenAI(api_key="")

def google_search(query):
    # Perform a Google search (this example uses scraping, which is against Google’s TOS)
    search_url = f"https://www.google.com/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    
    # Check if the request was successful
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('h3', limit=30)  # Get titles of the first 30 results
        return [result.get_text() for result in results if result.get_text()]
    else:
        print("Failed to retrieve results")
        return []

def generate_gpt_response(prompt, additional_info):
    combined_prompt = f"{prompt}\n\nAdditional Information:\n" + "\n".join(additional_info)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": combined_prompt}
            ]
        )
        return response.choices[0].message.content  # Correct way to access the content
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    user_prompt = input("Enter your prompt: ")
    additional_info = google_search(user_prompt)
    
    if additional_info:
        gpt_output = generate_gpt_response(user_prompt, additional_info)
        print("GPT Output:\n", gpt_output)
    else:
        print("No additional information found.")
