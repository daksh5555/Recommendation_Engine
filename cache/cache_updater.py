# cache_updater.py
import time
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from cache_manager import update_cache
from config import CACHE_EXPIRY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Initialize scheduler to update cache every `CACHE_EXPIRY` seconds
    scheduler = BackgroundScheduler()
    scheduler.add_job(update_cache, 'interval', seconds=CACHE_EXPIRY)
    scheduler.start()

    try:
        logger.info("Scheduler started. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)  # Sleep for 1 second, allowing scheduler to run
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down...")
        scheduler.shutdown()
        logger.info("Scheduler shut down.")
