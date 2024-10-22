# to run this:
# screen and stuff
# export CONSUMER_KEY= "pGBDoAaEpkliVKBOLwjtcmHGc"
# export CONSUMER_SECRET= "xF3g1wrP50b6BlZEd20u4oVfjgH1FGQcuWUzlQO5aUWOufvlhw"
# export ACCESS_TOKEN= "622518493-6VcLIPprbQbv9wkcBBPvCle8vsjU9fE85Dq9oStl"
# export ACCESS_TOKEN_SECRET= "tH9aKQbQQ1iRdYTcLSsPwitl44BkAc6jilrsU0ifnXvZhq"

import datetime
import time
import tweepy
import logging
import os
logger = logging.getLogger()


def create_api():
    consumer_key = os.getenv("CONSUMER_KEY")
    consumer_secret = os.getenv("CONSUMER_SECRET")
    access_token = os.getenv("ACCESS_TOKEN")
    access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")
    # consumer_key = API_KEY
    # consumer_secret = API_SECRET
    # access_token = ACCESS_TOKEN
    # access_token_secret = ACCESS_SECRET

    print('Creating and verifying api.')

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True, 
        wait_on_rate_limit_notify=True)

    api.verify_credentials()
    print('Credentials verified.')
    return api

def follow_followers(api):
    print('Following all followers.')
    logger.info("Retrieving and following followers")
    for follower in tweepy.Cursor(api.followers).items():
        if not follower.following:
            logger.info(f"Following {follower.name}")
            try:
                follower.follow()
            except:
                logger.error("Something went wrong while trying to follow", exc_info=True)

def human_weekday(w):
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    try:
        return weekdays[w]
    except:
        return ''

def get_hashtags(api):
    print('Retrieving hashtags.')
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

    wkd = human_weekday(datetime.datetime.today().weekday())
    hashtags = '\n#deepShowerThoughts #AI #MachineLearning #' + wkd + 'Thoughts'
    for entry in entries[:3]:
        hashtag = entry[0]
        hashtags += ' ' + hashtag
    return hashtags

def check_mentions(api, since_id):
    logger.info("Retrieving mentions")
    new_since_id = since_id
    for tweet in tweepy.Cursor(api.mentions_timeline,since_id=since_id).items():
        new_since_id = max(tweet.id, new_since_id)

        # change magick to something which generates a sentence
        prompt = tweet.text[15:]
        try:
            reply = generate_tweet(prompt=prompt)
        except:
            logger.error('failed to generate tweet', exc_info=True)

        if tweet.user.name == 'deepShowerThoughts':
            pass
        else:
            try:
                api.update_status(status = reply,
                    in_reply_to_status_id = tweet.id,
                    auto_populate_reply_metadata=True)
            except:
                logger.error("Failed to reply.", exc_info=True)
        
        try:
            tweet.favorite()
        except:
            logger.error("Failed to like tweet.", exc_info=True)

        if not tweet.user.following:
            try:
                tweet.user.follow()
            except:
                logger.error('could not follow', exc_info=True)

        # if tweet.in_reply_to_status_id is not None:
        #     continue

    return new_since_id

def generate_tweet(source='giacomo',prompt='', hashtags=''):
    print('Generating tweet.')

    root_path = os.path.join('ai', source)
    import pickle
    params_path = os.path.join(root_path, 'params.pkl')
    params = pickle.load(open(params_path,'rb'))

    model_path = os.path.join(root_path, 'model.pt')
    from ai.charmodel import charModel
    from ai.charvocabulary import charVocabulary
    from ai.charsample import gen_samp
    import torch
    
    dict_path = os.path.join(root_path, 'dict.pkl')
    token_to_idx = pickle.load(open(dict_path,'rb'))
    vocab = charVocabulary(token_to_idx=token_to_idx)

    model = charModel(**params)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    out = gen_samp(model=model, vocab=vocab, prompt=prompt) + hashtags
    out = out[:280]
    return out

def main():
    api = create_api()
    # change this?
    with open('sinceid.txt', 'r') as f:
        try:
            since_id = int(f.readline())
        except:
            since_id = 1178361132878307329
    print(since_id)
    while True:
        try:
            follow_followers(api)
        except:
            logger.error('failed to follow followers', exc_info=True)
        # hashtags = get_hashtags(api)
        # new_tweet = generate_tweet(hashtags=hashtags)
        # print('Updating twitter status.')
        # try:
        #     api.update_status(new_tweet)
        # except:
        #     logger.error("Something went wrong.", exc_info= True)

        print('updating since_id')
        try:
            since_id = check_mentions(api, since_id)
            with open('sinceid.txt', 'w') as f:
                f.write(str(since_id))
        except:
            logger.error('failed to run check_mentions', exc_info=True)

        logger.info("Waiting...")
        sec = 60
        print(f'Going to sleep for {sec} seconds.')
        time.sleep(sec) 

if __name__ == "__main__":
    main()