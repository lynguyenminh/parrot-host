import argparse
import requests
import json
from base64 import b64encode 


base_url = "https://api.joinparrot.ai/v1"
# base_url = "http://0.0.0.0:8000/v1"

def get_token():
    """
    This function performs addition of two numbers taken as command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Get Token for Parrot Host")
    parser.add_argument("--username", type=str, help="Your Parrot username")
    parser.add_argument("--password", type=str, help="Your Parrot password")

    args = parser.parse_args()

    url = f"{base_url}/user/login"
    # Authorization token: we need to base 64 encode it 
    # and then decode it to acsii as python 3 stores it as a byte string
    def basic_auth(username, password):
        token = b64encode(f"{username}:{password}".encode('utf-8')).decode("ascii")
        return f'Basic {token}'

    #then connect
    headers = { 'Authorization' : basic_auth(args.username, args.password) }        
    response = requests.post(url, headers= headers)

    # obtain the token
    token = response.json()["data"]["access_token"]
    print("YOUR TOKEN:\n", token)
    return args.username, token

def replace_text_in_file(source_file, target_file, old_text, new_text):
    """
    Replaces occurrences of old_text with new_text in a file and saves the result to a new file.

    Args:
    source_file: Path to the file to read from.
    target_file: Path to the file to write to.
    old_text: The text to be replaced.
    new_text: The replacement text.
    """
    try:
        with open(source_file, "r") as source:
            data = source.read()
            updated_data = data.replace(old_text, new_text)

        with open(target_file, "w") as target:
            target.write(updated_data)

        print(f"ENV file created successfully.")

    except FileNotFoundError:
        print(f"Error: ENV temaplte file '{source_file}' not found.")


if __name__ == "__main__":
    username, token = get_token()
    replace_text_in_file('.env_template', '.env_template', "<PUT_YOUR_USERNAME_HERE>", username)
    replace_text_in_file('.env_template', '.env', '<PUT_YOUR_TOKEN_HERE>', token)
