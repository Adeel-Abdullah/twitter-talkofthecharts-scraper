import os
from dotenv import load_dotenv
import asyncio
from datetime import datetime
from twikit import Client
import time
import json
import re
from dateutil import parser

# ----------------------
# CONFIG
# ----------------------

# Load environment variables from .env file
load_dotenv()

USERNAME = "talkofthecharts"
CUTOFF_DATE = datetime(2023, 8, 10)  # earliest chart to scrape
DOWNLOAD_DIR = "charts_images"
CHECKPOINT_FILE = "scraper_checkpoint.json"

# make sure folder exists
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# ----------------------
# Checkpoint functions
# ----------------------
def load_checkpoint():
    """Load the last processed tweet ID and date from checkpoint file"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                checkpoint = json.load(f)
                print(f"[checkpoint] Loaded checkpoint: {checkpoint}")
                return checkpoint.get('last_tweet_id'), checkpoint.get('last_tweet_date')
        except Exception as e:
            print(f"[checkpoint] Error loading checkpoint: {e}")
    return None, None

def save_checkpoint(tweet_id, tweet_date):
    """Save the current tweet ID and date to checkpoint file"""
    checkpoint = {
        'last_tweet_id': tweet_id,
        'last_tweet_date': tweet_date,
        'timestamp': datetime.now().isoformat()
    }
    try:
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        print(f"[checkpoint] Saved checkpoint: tweet_id={tweet_id}, date={tweet_date}")
    except Exception as e:
        print(f"[checkpoint] Error saving checkpoint: {e}")

# ----------------------
# Helper functions
# ----------------------
def extract_chart_date(text):
    """Extract chart date from tweet text like 'chart dated August 23rd, 2025'"""
    # Example: "Final Billboard Hot 100 Predictions (chart dated August 23rd, 2025)"
    match = re.search(r'chart dated ([A-Za-z]+ \d{1,2}[a-z]{2}, \d{4})', text)
    if match:
        date_str = match.group(1)
        # Remove ordinal suffix (st, nd, rd, th)
        date_str = re.sub(r'(\d{1,2})(st|nd|rd|th)', r'\1', date_str)
        try:
            dt = parser.parse(date_str)
            return dt.strftime('%Y-%m-%d')
        except Exception as e:
            print(f"[extract_chart_date] Failed to parse date: {date_str} ({e})")
    return None

# ----------------------
# Main scraper
# ----------------------
async def scrape_final_charts():

    # Get credentials from environment variables


    EMAIL = os.getenv("TWITTER_EMAIL")
    USERNAME_ENV = os.getenv("TWITTER_LOGIN_USERNAME")
    PASSWORD = os.getenv("TWITTER_PASSWORD")
    COOKIES_FILE = "twitter_cookies.json"

    # Rate limiting variables
    request_count = 0
    rate_limit_start = time.time()
    MAX_REQUESTS_PER_MINUTE = 10
    
    async def rate_limit():
        nonlocal request_count, rate_limit_start
        current_time = time.time()
        
        # Reset counter if a minute has passed
        if current_time - rate_limit_start >= 60:
            request_count = 0
            rate_limit_start = current_time
        
        # If we've hit the limit, wait until the minute is up
        if request_count >= MAX_REQUESTS_PER_MINUTE:
            sleep_time = 60 - (current_time - rate_limit_start)
            if sleep_time > 0:
                print(f"[rate_limit] Rate limit reached. Sleeping for {sleep_time:.2f} seconds...")
                await asyncio.sleep(sleep_time)
                request_count = 0
                rate_limit_start = time.time()
        
        request_count += 1
        print(f"[rate_limit] Request {request_count}/{MAX_REQUESTS_PER_MINUTE} in current minute")

    print(f"[scrape_final_charts] Starting with USERNAME={USERNAME}, CUTOFF_DATE={CUTOFF_DATE}")
    
    # Load checkpoint
    last_tweet_id, last_tweet_date = load_checkpoint()
    if last_tweet_id:
        print(f"[scrape_final_charts] Resuming from checkpoint: tweet_id={last_tweet_id}, date={last_tweet_date}")
    else:
        print(f"[scrape_final_charts] No checkpoint found, starting fresh")
    
    client = Client(language="en-US")
    # Try to load cookies if available
    if os.path.exists(COOKIES_FILE):
        print(f"[scrape_final_charts] Cookies file found: {COOKIES_FILE}")
        client.load_cookies(COOKIES_FILE)
        print(f"[scrape_final_charts] Loaded cookies from {COOKIES_FILE}")
        # Optionally, check if cookies are still valid by making a simple request
        try:
            print(f"[scrape_final_charts] Checking if cookies are valid...")
            await rate_limit()  # Apply rate limit
            await client.get_user_by_screen_name(USERNAME)
            print(f"[scrape_final_charts] Cookies are valid.")
        except Exception as e:
            print(f"[scrape_final_charts] Cookies invalid or expired: {e}\nLogging in again...")
            await rate_limit()  # Apply rate limit
            await client.login(auth_info_1=EMAIL, auth_info_2=USERNAME_ENV, password=PASSWORD)
            client.save_cookies(COOKIES_FILE)
            print(f"[scrape_final_charts] Saved new cookies to {COOKIES_FILE}")
    else:
        print(f"[scrape_final_charts] No cookies file found.")
        if not all([EMAIL, USERNAME_ENV, PASSWORD]):
            raise ValueError("Missing Twitter authentication info in .env file.")
        print(f"[scrape_final_charts] Logging in with credentials...")
        await rate_limit()  # Apply rate limit
        await client.login(auth_info_1=EMAIL, auth_info_2=USERNAME_ENV, password=PASSWORD)
        client.save_cookies(COOKIES_FILE)
        print(f"[scrape_final_charts] Saved cookies to {COOKIES_FILE}")

    # Build search query for specific tweets
    search_query = f'"Final Billboard Hot 100 Predictions" from:{USERNAME}'
    
    # Add date range using both cutoff and checkpoint dates
    if CUTOFF_DATE:
        cutoff_str = CUTOFF_DATE.strftime('%Y-%m-%d')
        # search_query += f' since:{cutoff_str}'
        # search_query += f' since:{cutoff_str}'  # Comment out if not needed
    
    # Use checkpoint date as end date if available
    if last_tweet_date:
        try:
            # Parse the checkpoint date and format it for Twitter search
            checkpoint_dt = datetime.fromisoformat(last_tweet_date.replace('Z', '+00:00'))
            if checkpoint_dt.tzinfo:
                checkpoint_dt = checkpoint_dt.replace(tzinfo=None)
            until_str = checkpoint_dt.strftime('%Y-%m-%d')
            search_query += f' until:{until_str}'
            print(f"[scrape_final_charts] Using checkpoint date as end date: {until_str}")
        except Exception as e:
            print(f"[scrape_final_charts] Error parsing checkpoint date: {e}")
        
    # Add filter for images to make it more efficient
    search_query += ' filter:images'
    
    print(f"[scrape_final_charts] Searching with query: {search_query}")
    await rate_limit()  # Apply rate limit
    tweets = await client.search_tweet(search_query, product='Latest')
    
    total_fetched = 0
    while True:
        print(f"[scrape_final_charts] Got {len(tweets)} tweets.")
        for tweet in tweets:
            print(f"[scrape_final_charts] Processing tweet: {tweet.created_at}")
            print(f"[scrape_final_charts] Tweet content: {getattr(tweet, 'full_text', '')}")
            
            # parse date
            tweet_date = datetime.strptime(tweet.created_at, "%a %b %d %H:%M:%S %z %Y").replace(tzinfo=None)
            print(f"[scrape_final_charts] Parsed tweet_date: {tweet_date}")

            # stop when we reach cutoff
            if tweet_date < CUTOFF_DATE:
                print(f"[scrape_final_charts] Reached cutoff date, stopping scrape.")
                return

            # All tweets from search should be new (not processed before)
            # since we used checkpoint date as 'until' parameter
            print(f"[scrape_final_charts] Found chart for {tweet_date.date()}")
            
            # Extract chart date from tweet text
            chart_date_str = extract_chart_date(tweet.full_text)
            if chart_date_str:
                chart_date_for_filename = chart_date_str
                print(f"[scrape_final_charts] Extracted chart date: {chart_date_str}")
            else:
                chart_date_for_filename = str(tweet_date.date())
                print(f"[scrape_final_charts] Using tweet date as fallback: {chart_date_for_filename}")
            
            # download images if present
            if tweet.media:
                print(f"[scrape_final_charts] Tweet has {len(tweet.media)} media items.")
                for i, m in enumerate(tweet.media, start=1):
                    media_type = getattr(m, 'type', None)
                    media_url = getattr(m, 'url', None)
                    print(f"[scrape_final_charts] Media item {i}: type={media_type}, url={media_url}")
                    if media_type == "photo" and media_url:
                        filename = f"{chart_date_for_filename}_{i}.jpg"
                        filepath = os.path.join(DOWNLOAD_DIR, filename)
                        
                        # Check if file already exists
                        if not os.path.exists(filepath):
                            print(f"[scrape_final_charts] Downloading new image: {filename}")
                            await rate_limit()  # Apply rate limit before download
                            await m.download(output_path=filepath)
                        else:
                            print(f"[scrape_final_charts] Image already exists: {filename}")
                    elif media_type == "photo" and not media_url:
                        print(f"[scrape_final_charts] Skipping image: media_url_https is None for media item {i}")
            
            # Save checkpoint after processing a tweet with images
            if tweet.media:
                save_checkpoint(tweet.id, tweet_date.isoformat())
                    
        total_fetched += len(tweets)
        print(f"[scrape_final_charts] Total tweets processed so far: {total_fetched}")
        # Pagination: fetch next batch
        if hasattr(tweets, 'next') and callable(getattr(tweets, 'next')):
            print(f"[scrape_final_charts] Fetching next page of tweets...")
            await rate_limit()  # Apply rate limit before pagination
            tweets = await tweets.next()
            if not tweets or len(tweets) == 0:
                print(f"[scrape_final_charts] No more tweets to fetch.")
                break
        else:
            print(f"[scrape_final_charts] No pagination method found, stopping.")
            break

# ----------------------
# Run it
# ----------------------
if __name__ == "__main__":
    asyncio.run(scrape_final_charts())
