#Importing libraries
import pandas as pd
from   google_drive_downloader import GoogleDriveDownloader as gdd
import pygsheets
import re
import requests
import spacy
from spacy.tokenizer import _get_regex_pattern
import contractions
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from math import sqrt
from ast import literal_eval
import numpy as np
import gradio as gr
import os

# Initiallization

#Downloading necessary spacy models
try:
        nlp = spacy.load('en_core_web_md')
except:
    spacy.cli.download('en_core_web_md')
    nlp = spacy.load('en_core_web_md')

#initiating bearer token
bearer_token = os.environ['bearer_token']
#Retrieving the tweet db for comparision

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

#selecting specific sheets
Tweet_sheet_old            = google_sheet.worksheet_by_title('Htag Recom tweets')
Tweet_Db_main              = Tweet_sheet_old.get_as_df()

#Defining functions 

# Function to fetch necessary user info
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
def clean_tweet(new_tweet):
    """Function to clean the tweet text entered"""
    #cleaning the tweet
    new_tweet_cleaned= ' '.join(re.sub("([^@_#'.!?0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",new_tweet).split())
    #cleaning the text again. This time removing the trailing hashtags or removing # symbol from tweet body. We also replace @ mentions with the associated display names
    new_tweet_cleaned2= Remove_trailing_hashtags_and_replacing_usernames(new_tweet_cleaned)
    #cleaning the text again. This time fixing the contractions
    new_tweet_cleaned3= contractions.fix(new_tweet_cleaned2)
    return new_tweet_cleaned3

def hashtag_generator(Tweet,hashtag_count):
    """Function that will generate hashtags for the entered text"""
    
    # Computing additional columns and similarity scores

    compare_DB = Tweet_Db_main.copy() #working on a copy
    compare_DB = compare_DB[compare_DB['Hashtags'].notnull()] #removing any nulls

    #cleaning the entered tweet text
    new_tweet_cleaned3 = clean_tweet(Tweet)

    #computing cosine similarity
    TfIdf_cos_similarity = []

    for tweet in compare_DB['Tweet Text cleaned']:
        """Computing TF_IDF cosine similarity"""
        similarity_list = [new_tweet_cleaned3]+[tweet]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(similarity_list)
        arr = X.toarray()

        TfIdf_cos_similarity.append(cosine_similarity(arr)[0,1])
    #creating a column for cosine similarity
    compare_DB['Cosine Similarity'] = TfIdf_cos_similarity

    # Creating a new row for each hashtag and removing duplicated rows
    compare_DB['Hashtags'] = compare_DB['Hashtags'].apply(literal_eval) #convert to list type
    compare_DB_expanded    = compare_DB.explode('Hashtags').drop_duplicates(keep='first').reset_index(drop=True)
    #Computing user influence
    compare_DB_expanded['Avg Verified Status'] = compare_DB_expanded.groupby(['Hashtags'])['Verified Status Num'].transform('mean')
    compare_DB_expanded['Avg Follower Count']  = compare_DB_expanded.groupby(['Hashtags'])['Followers'].transform('mean')
    #setting parameters
    alpha = 1
    beta  = 0.25
    compare_DB_expanded['Influence Score']     = alpha * compare_DB_expanded['Avg Verified Status'] + beta * np.log(compare_DB_expanded['Avg Follower Count']+1)

    #computing hashtag frequency
    compare_DB_expanded['Hashtag Freq']  = compare_DB_expanded.groupby(['Hashtags'])['Followers'].transform('count')/compare_DB_expanded.shape[0]

    # #Evaluating the cut off values of scores (done initially to find optimum cutt off points. Commenting out rather than deleting for future reference)
    # compare_DB_expanded['Influence Score'].describe()
    # compare_DB_expanded['Cosine Similarity'].describe()
    # compare_DB_expanded['Hashtag Freq'].describe()
    # compare_DB_expanded[compare_DB_expanded['Cosine Similarity'].apply(lambda x: True if (x >= 0.3) else False)]['Hashtags'].unique()


    #computing recommendation scores (RS)

    compare_DB_expanded['RS Cosine']    = compare_DB_expanded['Cosine Similarity'].apply(lambda x: 1 if (x >= 0.3) else 0)
    compare_DB_expanded['RS Influence'] = compare_DB_expanded['Influence Score'].apply(lambda x: 1 if (x >= 4.1) else 0)
    compare_DB_expanded['RS Frequency'] = compare_DB_expanded['Hashtag Freq'].apply(lambda x: 1 if (x >= 0.001) else 0)
    # generating hashtags to recommend
    compare_DB_expanded['compound score'] = compare_DB_expanded['Cosine Similarity']*compare_DB_expanded['Influence Score']
    candidate_hashtags = compare_DB_expanded[(compare_DB_expanded['RS Cosine']+compare_DB_expanded['RS Influence']+compare_DB_expanded['RS Frequency'])>1].sort_values(by=['compound score'])['Hashtags'].str.lower().drop_duplicates(keep='first').reset_index(drop=True)
    # Subsetting for top 10 or lesser hashtags among candidates
    if len(candidate_hashtags)>hashtag_count:
        recommended_hashtags = candidate_hashtags[0:hashtag_count]
    else:
        recommended_hashtags = candidate_hashtags
    # Recommending relevant hashtags to users

    htag_list = "The hashtags recommended for entered text are:"

    if len(recommended_hashtags)==0:
        print("Sorry no suggestions generated.")
        htag_list = ""
    else:
        for htag in recommended_hashtags:
            htag_list += " #"+htag

    return(htag_list)

# Wrapping recommender function around gradio wrapper
htag_recommender = gr.Interface(fn      = hashtag_generator,
                                inputs  = [gr.inputs.Textbox(lines = 10, placeholder = "Enter the tweet here...."),gr.inputs.Slider(1,15,step=1,label="Maximum number of recomended hashtags")],
                                outputs = "text",
                                allow_flagging = "never",
                                title = "Hashtag recommendation engine"
                                )

#Initializing Gradio interface
htag_recommender.launch(auth = ('SttAdmin','Hashtag123'))