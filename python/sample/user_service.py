"""Sample file to talk to (uv run --extra gguf python -m voice sample/user_service.py).

Try: "add a docstring to parse config", "rename get user to fetch user", "add a method to the cache that removes
a single user", "log a warning when the request fails", "actually, never mind".
"""

import json
import logging
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

TIMEOUT = 10
CACHE_SECONDS = 60


def parse_config(path):
    with open(path) as f:
        data = json.load(f)
    return {"url": data["url"], "retries": data.get("retries", 3)}


class UserCache:
    def __init__(self, max_age=CACHE_SECONDS):
        self.max_age = max_age
        self._users = {}

    def get(self, user_id):
        entry = self._users.get(user_id)
        if entry is None:
            return None
        user, stored_at = entry
        if time.time() - stored_at > self.max_age:
            del self._users[user_id]
            return None
        return user

    def put(self, user_id, user):
        self._users[user_id] = (user, time.time())

    def clear(self):
        self._users.clear()


def get_user(base_url, user_id, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{base_url}/users/{user_id}", timeout=TIMEOUT) as response:
                return json.load(response)
        except urllib.error.URLError:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)


def format_user(user):
    return f"{user['name']} <{user['email']}>"


def show_user(config_path, user_id, cache=None):
    config = parse_config(config_path)
    user = cache.get(user_id) if cache else None
    if user is None:
        user = get_user(config["url"], user_id, config["retries"])
        if cache:
            cache.put(user_id, user)
    print(format_user(user))


if __name__ == "__main__":
    import sys

    show_user(sys.argv[1], sys.argv[2], UserCache())
