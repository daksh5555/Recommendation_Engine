# cache_manager.py
import requests
import redis
import json
import logging
from config import DATA_SOURCE_URL, CACHE_EXPIRY, REDIS_HOST, REDIS_PORT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Redis for caching
cache = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)

def fetch_data_from_api():
    try:
        # Fetch data from API
        users_response = requests.get(f'{DATA_SOURCE_URL}/search?type=user&query=sac')
        users_response.raise_for_status()
        users_data = users_response.json()

        feed_response = requests.get(f'{DATA_SOURCE_URL}/feed?page=1')
        feed_response.raise_for_status()
        feed_data = feed_response.json()

        # Cache the results with expiration time from config
        cache.set('users_data', json.dumps(users_data), ex=CACHE_EXPIRY)
        cache.set('feed_data', json.dumps(feed_data['posts']), ex=CACHE_EXPIRY)

        logger.info("New data fetched and cached successfully.")
        return True
    except requests.RequestException as e:
        logger.error(f"Error fetching data: {e}")
        return False

def update_cache():
    logger.info("Updating cache...")
    data_fetched = fetch_data_from_api()
    if not data_fetched:
        logger.error("Failed to update cache.")
    else:
        logger.info("Cache updated successfully.")
