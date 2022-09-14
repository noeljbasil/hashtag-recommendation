# %% [markdown]
# ##### STT Hashtag recommendation Tweet database scarpping and upload

# %%
#Importing Libraries

from   google_drive_downloader import GoogleDriveDownloader as gdd
import pygsheets
import snscrape.modules.twitter as sntwitter
from   datetime import date
from   dateutil.relativedelta import relativedelta
import pandas as pd
import re
import math
import requests
import tweepy
import spacy
from spacy.tokenizer import _get_regex_pattern
import contractions
import time
import regex as re

# %%
#Downloading necessary spacy models
try:
        nlp = spacy.load('en_core_web_sm')
except:
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# %%
#initiating tweepy client
bearer_token = os.environ['bearer_token']
tweepy_client = tweepy.Client(bearer_token=bearer_token)

# %%
#Initializing google drive parameters
gdrive_id =  os.environ['gdrive_id']

gdd.download_file_from_google_drive(file_id=gdrive_id,
                                    dest_path='./secret_key.json',
                                    unzip=True)

#authenticating with google sheets with pygsheets
client = pygsheets.authorize(service_account_file='secret_key.json')

#open google sheet
gsheet_key = os.environ['gsheet_key']
google_sheet = client.open_by_key(gsheet_key)

# %%
#Tweet Scrapping

def get_latest_tweets(keyword):
    """This function scraps tweets with #humantrafficking for the last 6 months"""

    # creating list to append tweet data to
    hashtag_tweets_list = []

    # using TwitterHashtagScraper to scrape tweets for a specific hashtag from last 6 months and append it to list
    for i,tweet in enumerate(sntwitter.TwitterHashtagScraper(f'#{keyword} since:{(date.today() - relativedelta(months=6)).strftime("%Y-%m-%d")}').get_items()):
        hashtag_tweets_list.append([tweet.content, tweet.hashtags, tweet.lang, tweet.user, tweet.id, tweet.date])

    # creating a dataframe from the tweets list above
    hashtags_tweets_df = pd.DataFrame(hashtag_tweets_list, columns=['Text', 'Hashtags', 'Language','User','Id','Date'])

    # subsetting for english tweets
    hashtags_tweets_df = hashtags_tweets_df[hashtags_tweets_df['Language']=='en'].drop('Language', axis=1)

    return hashtags_tweets_df

print("\n================================")
print("Tweets collection started")
print("================================\n")

HT_hashtags_tweets_df = get_latest_tweets('humantrafficking')
MS_hashtags_tweets_df = get_latest_tweets('modernslavery')
print("\n================================")
print("Tweets collection completed")
print("================================\n")

#Join both dataframes and resetting index
tweet_df = pd.concat([HT_hashtags_tweets_df,MS_hashtags_tweets_df],axis=0).reset_index(drop=True)

# %%
#Checking the ids of external tweets against cloud file to reduce redundant runs for data already collected

#selecting specific sheets
ext_twt_ids            = google_sheet.worksheet_by_title('External Tweet Ids')
ExtID_main             = ext_twt_ids.get_as_df()

#getting the ids from current run
try:
    currExtID              = tweet_df[['Id','Date']]
except:
    currExtID              = pd.DataFrame({'Id':[],'Date':[]})  


#subsetting external tweets for only tweets not seen in previous code runs
try:
    tweet_df_filtered = tweet_df[[id not in list(ExtID_main['Id']) for id in list(currExtID['Id'])]][['Text', 'Hashtags', 'User']]
except:
    tweet_df_filtered = tweet_df[['Text', 'Hashtags', 'User']]#no ids present in cloud as reference so not updating the tweet df table 

#Updating the reference table
try:
    ExtTwtId_expanded             = pd.concat([ExtID_main,currExtID],ignore_index=True).drop_duplicates(ignore_index=True)

    ExtTwtId_expanded['Date_new'] = ExtTwtId_expanded['Date'].apply(lambda x: x.date())
    ExtTwtId_expanded.drop('Date', axis=1,inplace=True)
    ExtTwtId_expanded.rename(columns = {'Date_new':'Date'},inplace=True)

    ExtTwtId_expanded_latest      = ExtTwtId_expanded[ExtTwtId_expanded['Date']>=(date.today() - relativedelta(months=6))]

except:
    ExtTwtId_expanded             = currExtID.drop_duplicates(ignore_index=True)
    
    ExtTwtId_expanded['Date_new'] = ExtTwtId_expanded['Date'].apply(lambda x: x.date())
    ExtTwtId_expanded.drop('Date', axis=1,inplace=True)
    ExtTwtId_expanded.rename(columns = {'Date_new':'Date'},inplace=True)
    
    ExtTwtId_expanded_latest      = ExtTwtId_expanded[ExtTwtId_expanded['Date']>=(date.today() - relativedelta(months=6))]

ExtID_main_updated = ExtTwtId_expanded[['Id','Date']]

#clearing existing values from the sheets
ext_twt_ids.clear(start='A1', end=None, fields='*')

#writing dataframes into the sheets
ext_twt_ids.set_dataframe(ExtID_main_updated, start=(1,1))

# %%
#cleaning the text
tweet_df_filtered['User ID'] = tweet_df_filtered['User'].apply(lambda x: ' '.join(re.sub("https://twitter.com/"," ",str(x)).split()))
tweet_df_filtered.drop('User', axis=1,inplace=True)
tweet_df_filtered['Cleaned Text']= tweet_df_filtered['Text'].apply(lambda x: ' '.join(re.sub("([^@_#'.!?0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x).split()))
#Removing redundant column and resetting index
tweet_df_filtered.drop('Text', axis=1,inplace=True)
tweet_df_filtered.reset_index(drop=True,inplace=True)

# %%
#Defining functions to fetch necessary user info
def create_url(user_names_list, user_fields):
    user_names = ','.join(user_names_list) if len(user_names_list)>1 else user_names_list[0]
    usernames = f"usernames={user_names}"
    url = "https://api.twitter.com/2/users/by?{}&{}".format(usernames, user_fields)
    return url

def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2UserLookupPython"
    return r

def connect_to_endpoint(url):
    response = requests.request("GET", url, auth=bearer_oauth,)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()

def get_user_info(list_of_user_names):
    
    followers   = []
    verified    = []
    user_handle = []
    
    user_fields  = "user.fields=verified,public_metrics"
    url = create_url(list_of_user_names,user_fields)
    json_response = connect_to_endpoint(url)
   
    for user in json_response['data']: #for valid users whose data is returned
        try:
            verified.append(user['verified'])
        except:
            verified.append("False")
        try:
            followers.append(user['public_metrics']['followers_count'])
        except:
            followers.append(0)
        user_handle.append(user['username'])
    if 'errors' in list(json_response.keys()):
        for user in json_response['errors']: #for invalid users
            followers.append(0)
            verified.append("False")
            user_handle.append(user["value"])
    return followers,verified,user_handle

def get_display_name(list_of_user_names):
     
    user_fields  = "user.fields=name,username"
    url = create_url(list_of_user_names,user_fields)
    json_response = connect_to_endpoint(url)

    for user in json_response['data']: #for valid users whose data is returned
        try:
            display_name = user['name']
        except:
            display_name = re.findall("@([a-zA-Z0-9_]{1,50})",user['username'])[0]

    if 'errors' in list(json_response.keys()):
        for user in json_response['errors']: #for invalid users
            display_name = user["value"]
    return display_name

# %%
# Defining function to clean up hashtag and mentions in tweet body
def Remove_trailing_hashtags_and_replacing_usernames (tweet):
    """Funtion to remove trailing hashtags or remove # symbols from body of tweet. This function also replaces @ mentions with the respective usernames"""
    # get default pattern for tokens that don't get split
    re_token_match = _get_regex_pattern(nlp.Defaults.token_match)
    # add your patterns (here: hashtags and in-word hyphens)
    re_token_match = f"({re_token_match}|#\w+|\w+-\w+)"

    # overwrite token_match function of the tokenizer
    nlp.tokenizer.token_match = re.compile(re_token_match).match
    doc = nlp(tweet)
    tweet_cleaned = ""
    for token in doc:
        if bool(re.findall("@([a-zA-Z0-9_]{1,50})", token.text)): #check if it is a @ mention
            try:
                tweet_cleaned=tweet_cleaned+" "+get_display_name(re.findall("@([a-zA-Z0-9_]{1,50})", token.text)) #replacing @ with user name
            except:
                tweet_cleaned=tweet_cleaned+" "+token.text
        else:
            if token.text == str(doc[0]): #check if it is the first word
                if bool(re.findall("#([a-zA-Z0-9_]{1,50})", token.text)): #check if it is a hashtag

                    if len(re.findall('([A-Z][^A-Z]*)', token.text))>1 and not(token.text.isupper()):
                        updated_word=""
                        for sub_word in re.findall('([A-Z][^A-Z]*)', token.text):
                            if updated_word=="":
                                updated_word+=sub_word
                            else:
                                updated_word= updated_word+" "+sub_word
                    elif len(re.sub("_"," ",token.text).split())>1:
                        updated_word=""
                        for sub_word in re.sub("_"," ",token.text).split():
                            if updated_word=="":
                                updated_word+=sub_word
                            else:
                                updated_word= updated_word+" "+sub_word
                    else:
                        updated_word = re.findall("#([a-zA-Z0-9_]{1,50})", token.text)[0]
                    
                    tweet_cleaned=tweet_cleaned+" "+updated_word

                else:
                    tweet_cleaned=tweet_cleaned+" "+token.text
            else:
                if bool(re.findall("#([a-zA-Z0-9_]{1,50})", token.text)): #check if it is a hashtag
                    if token.nbor(-1).pos_ in  ['SCONJ' ,'PART', 'DET', 'CCONJ', 'CONJ' ,'AUX', 'ADP', 'ADJ', 'VERB' ,'INTJ' ,'PRON', 'ADV']: #check pos of previous word
                        
                        if len(re.findall('([A-Z][^A-Z]*)', token.text))>1 and not(token.text.isupper()):
                            updated_word=""
                            for sub_word in re.findall('([A-Z][^A-Z]*)', token.text):
                                if updated_word=="":
                                    updated_word+=sub_word
                                else:
                                    updated_word= updated_word+" "+sub_word
                        elif len(re.sub("_"," ",token.text).split())>1:
                            updated_word=""
                            for sub_word in re.sub("_"," ",token.text).split():
                                if updated_word=="":
                                    updated_word+=sub_word
                                else:
                                    updated_word= updated_word+" "+sub_word
                        else:
                            updated_word = re.findall("#([a-zA-Z0-9_]{1,50})", token.text)[0]
                        
                        tweet_cleaned=tweet_cleaned+" "+updated_word
                    else:
                        pass #remove hashtag
                else:
                    tweet_cleaned=tweet_cleaned+" "+token.text 
    return  tweet_cleaned

# %%
#cleaning the text again. This time removing the trailing hashtags or removing # symbol from tweet body. We also replace @ mentions with the associated display names
tweet_df_filtered['Tweet Text']= tweet_df_filtered['Cleaned Text'].apply(lambda x: Remove_trailing_hashtags_and_replacing_usernames(x))
#Removing redundant column and resetting index
tweet_df_filtered.drop('Cleaned Text', axis=1,inplace=True)
tweet_df_filtered.reset_index(drop=True,inplace=True)

# %%
#cleaning the text again. This time fixing the contractions
tweet_df_filtered['Tweet Text cleaned']= tweet_df_filtered['Tweet Text'].apply(lambda x: contractions.fix(x))
#Removing redundant column and resetting index
tweet_df_filtered.drop('Tweet Text', axis=1,inplace=True)
tweet_df_filtered.reset_index(drop=True,inplace=True)

# %%
#Sleeping for 2 hours to overcome rate limit
time.sleep(60*60*2)

# %%
#Getting unique user ids to retrieve user info
unique_user_ids = list(tweet_df_filtered['User ID'].unique())
start_index = 0
end_index   = 100
followers   = []
verified    = []
user_handle = []
for iteration in range(math.ceil(len(unique_user_ids)/100)):
    print(f"Iteration {iteration+1}/{math.ceil(len(unique_user_ids)/100)}")
    list_of_users                  = unique_user_ids[start_index:end_index]
    f,v,u = get_user_info(list_of_users)
    followers+=f
    verified+=v
    user_handle+=u
    start_index+=100
    if end_index+100<=len(unique_user_ids):
        end_index+=100
    else:
        end_index=len(unique_user_ids)

unique_user_info = pd.DataFrame({'User ID':unique_user_ids,'Followers':followers,'Verified Status':verified})


tweets_df_with_usernames = tweet_df_filtered.merge(unique_user_info,on='User ID',how='left')

#converting boolean into 0 and 1 for calculating influence score later on
tweets_df_with_usernames['Verified Status Num'] = tweets_df_with_usernames['Verified Status'].apply(lambda x: 1 if (str(x) == "True") else 0)

#removing redundant column
tweets_df_with_usernames.drop('Verified Status', axis=1,inplace=True)

# %%
#Getting the tweets by STT
text                            = []
hashtags                        = []
follower_count,dummy,userid     = get_user_info(['STOPTHETRAFFIK'])
tweet_id                        = []

for tweet in tweepy.Paginator(tweepy_client.get_users_tweets,id=28075780,
                              tweet_fields=['entities'],
                              max_results=100).flatten():
    text.append(tweet.text)
    
    try:
        htag_list=[]
        for htag in tweet.entities['hashtags']:
            htag_list.append(htag['tag'])
    except:
        htag_list=[]
    hashtags.append(htag_list)

    tweet_id.append(tweet.id)

STT_tweets_df = pd.DataFrame({'User ID':list(userid)*len(text),'Text':text,'Followers':list(follower_count)*len(text), 'Hashtags':hashtags,'Verified Status Num':[1]*len(text),'Id':tweet_id})

# %%
#Checking the ids of external tweets against cloud file to reduce redundant runs for data already collected

#selecting specific sheets
stt_twt_ids            = google_sheet.worksheet_by_title('STT Tweet Ids')
SttID_main             = stt_twt_ids.get_as_df()

#getting the ids from current run
try:
    currSttID              = STT_tweets_df[['Id']]
except:
    currSttID              = pd.DataFrame({'Id':[]})  

#subsetting external tweets for only tweets not seen in previous code runs
try:
    STT_tweets_df = STT_tweets_df[[id not in list(SttID_main['Id']) for id in list(currSttID['Id'])]][['User ID', 'Text', 'Followers','Hashtags','Verified Status Num']]
except:
    STT_tweets_df = STT_tweets_df[['User ID', 'Text', 'Followers','Hashtags','Verified Status Num']] #no ids present in cloud as reference so not updating the tweet df table 

#Updating the reference table
try:
    SttTwtId_expanded = pd.concat([SttID_main,currSttID],ignore_index=True).drop_duplicates(ignore_index=True)
    
except:
    SttTwtId_expanded = currSttID.drop_duplicates(ignore_index=True)

#clearing existing values from the sheets
stt_twt_ids.clear(start='A1', end=None, fields='*')

#writing dataframes into the sheets
stt_twt_ids.set_dataframe(SttTwtId_expanded, start=(1,1))

# %%
#calculating the number of hashtags in each tweet
STT_tweets_df['Number of Hashtags'] = STT_tweets_df['Hashtags'].apply(lambda x: len(x))
#Filtering for tweets with hashtags & removing redundant column
STT_tweets_df_filtered = STT_tweets_df[STT_tweets_df['Number of Hashtags']!=0]
STT_tweets_df_filtered.drop('Number of Hashtags', axis=1,inplace=True)
STT_tweets_df_filtered.reset_index(drop=True,inplace=True)
#cleaning the text
STT_tweets_df_filtered['Cleaned Text']= STT_tweets_df_filtered['Text'].apply(lambda x: ' '.join(re.sub("([^@_#'.!?0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x).split()))
#Removing redundant column and resetting index
STT_tweets_df_filtered.drop('Text', axis=1,inplace=True)
STT_tweets_df_filtered.reset_index(drop=True,inplace=True)
#cleaning the text again. This time removing the trailing hashtags or removing # symbol from tweet body. We also replace @ mentions with the associated display names
STT_tweets_df_filtered['Tweet Text']= STT_tweets_df_filtered['Cleaned Text'].apply(lambda x: Remove_trailing_hashtags_and_replacing_usernames(x))
#Removing redundant column and resetting index
STT_tweets_df_filtered.drop('Cleaned Text', axis=1,inplace=True)
STT_tweets_df_filtered.reset_index(drop=True,inplace=True)
#cleaning the text again. This time fixing the contractions

STT_tweets_df_filtered['Tweet Text cleaned']= STT_tweets_df_filtered['Tweet Text'].apply(lambda x: contractions.fix(x))
#Removing redundant column and resetting index
STT_tweets_df_filtered.drop('Tweet Text', axis=1,inplace=True)
STT_tweets_df_filtered.reset_index(drop=True,inplace=True)

# %%
#Merging both the tweet dataframes together to get the final Tweets DB
Htag_Tweet_DB = pd.concat([tweets_df_with_usernames[['User ID','Tweet Text cleaned','Followers','Hashtags', 'Verified Status Num']],STT_tweets_df_filtered[['User ID','Tweet Text cleaned','Followers','Hashtags', 'Verified Status Num']]],ignore_index=True)

# %%
#Adding the new tweets to the main data created for previous runs

#selecting specific sheets
Tweet_sheet_old            = google_sheet.worksheet_by_title('Htag Recom tweets')
Tweet_Db_main              = Tweet_sheet_old.get_as_df()

Tweet_Db_main_updated = pd.concat([Tweet_Db_main,Htag_Tweet_DB],ignore_index=True)

#clearing existing values from the sheets
Tweet_sheet_old.clear(start='A1', end=None, fields='*')

#writing dataframes into the sheets
Tweet_sheet_old.set_dataframe(Tweet_Db_main_updated, start=(1,1))

# %%
# Creating a new row for each hashtag and removing duplicated rows
htags      = Tweet_Db_main_updated.apply(lambda x: pd.Series(x['Hashtags']),axis=1).stack().reset_index(level=1, drop=True)
htags.name = 'Tweet Recommended Hashtag'
Htag_DB    = Tweet_Db_main_updated.drop('Hashtags', axis=1).join(htags).drop_duplicates(keep='first').reset_index(drop=True)