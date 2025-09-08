import os
from dotenv import load_dotenv
import asyncio
from twikit import Client

# Load environment variables from .env file
load_dotenv()

EMAIL = os.getenv("TWITTER_EMAIL")
USERNAME_ENV = os.getenv("TWITTER_LOGIN_USERNAME")
PASSWORD = os.getenv("TWITTER_PASSWORD")
COOKIES_FILE = "twitter_cookies.json"

async def login_and_save_cookies():
    if not all([EMAIL, USERNAME_ENV, PASSWORD]):
        raise ValueError("Missing Twitter authentication info in .env file.")

    client = Client(language="en-US")
    await client.login(auth_info_1=EMAIL, auth_info_2=USERNAME_ENV, password=PASSWORD)
    # Save cookies to file
    client.save_cookies(COOKIES_FILE)
    print(f"Cookies saved to {COOKIES_FILE}")

if __name__ == "__main__":
    asyncio.run(login_and_save_cookies())
