# to run this:
# screen and stuff
# export CONSUMER_KEY= "pGBDoAaEpkliVKBOLwjtcmHGc"
# export CONSUMER_SECRET= "xF3g1wrP50b6BlZEd20u4oVfjgH1FGQcuWUzlQO5aUWOufvlhw"
# export ACCESS_TOKEN= "622518493-6VcLIPprbQbv9wkcBBPvCle8vsjU9fE85Dq9oStl"
# export ACCESS_TOKEN_SECRET= "tH9aKQbQQ1iRdYTcLSsPwitl44BkAc6jilrsU0ifnXvZhq"

# testing: !!!

import time
import tweepy
import logging
import os
logger = logging.getLogger()

def create_api():
    # consumer_key = os.getenv("CONSUMER_KEY")
    # consumer_secret = os.getenv("CONSUMER_SECRET")
    # access_token = os.getenv("ACCESS_TOKEN")
    # access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")
    consumer_key = API_KEY
    consumer_secret = API_SECRET
    access_token = ACCESS_TOKEN
    access_token_secret = ACCESS_SECRET

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True, 
        wait_on_rate_limit_notify=True)
    try:
        api.verify_credentials()
    except Exception as e:
        logger.error("Error creating API", exc_info=True)
        raise e
    logger.info("API created")
    return api

def follow_followers(api):
    logger.info("Retrieving and following followers")
    for follower in tweepy.Cursor(api.followers).items():
        if not follower.following:
            logger.info(f"Following {follower.name}")
            follower.follow()

def get_hashtags(api):
    location_tags = {'London': 44418, 'New York': 2459115, 'San Francisco': 2487956, 'Seattle': 2490383}
    popular_hashtags = {}
    for location_id in location_tags.values():
        geographic_trends = api.trends_place(location_id)[0]['trends']
        for trend in geographic_trends:
            if trend['name'] and trend['tweet_volume']:
                name = trend['name']
                volume = trend['tweet_volume']
                if name[0] == '#' and (name not in popular_hashtags):
                    popular_hashtags[name] = volume
    entries = sorted(popular_hashtags.items(), key=lambda item: item[1], reverse=True)

    # augment this with #DayOfWeekThoughts
    # import datetime
    hashtags = '\n#deepShowerThoughts #AI #MachineLearning #BigData'
    for entry in entries[:4]:
        hashtag = entry[0]
        hashtags += ' ' + hashtag
    return hashtags

def check_mentions_fav_reply(api, since_id):
    logger.info("Retrieving mentions")
    new_since_id = since_id
    for tweet in tweepy.Cursor(api.mentions_timeline,since_id=since_id).items():
        new_since_id = max(tweet.id, new_since_id)

        # change magick to something which generates a sentence
        seed = text[15:]
        if tweet.user == 'johntweetsthings':
            reply = magick(seed)
            api.update_status(status = reply,
                in_reply_to_status_id = tweet.id,
                auto_populate_reply_metadata=True)
        
        # does not seem to always work
        try:
            tweet.favorite()
        except:
            print(f"Fav did not work for tweet {tweet.id}.")

        if not tweet.user.following:
            tweet.user.follow()

        if tweet.in_reply_to_status_id is not None:
            continue

    return new_since_id

def tweet_something():
    # gen_samp


def main():
    api = create_api()
    # change this?
    since_id = 1
    while True:
        follow_followers(api)
        since_id = check_mentions(api, since_id)   
        logger.info("Waiting...")
        time.sleep(1800) 



if __name__ == "__main__":
    main()