
                                                           
import sys
from config import *
import pandas as pd
import numpy as np
from sklearn import datasets # sklearn comes with some toy datasets to practise
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
import spotipy
import json
import sys
from termcolor import cprint

from spotipy.oauth2 import SpotifyClientCredentials
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id,
                                                           client_secret=client_secret))

def get_uri(sp, title, artist):
    results=sp.search(q="track:"+title+" artist:"+artist,limit=5)
    if results['tracks']['items']==[]:   
        return None
    else:
        for i in range(len(results['tracks']['items'])):
            if (artist==results['tracks']['items'][i]['album']['artists'][0]['name']) and (title==results['tracks']['items'][i]['name']):
                return results['tracks']['items'][i]['uri']  
            
        return None
    
    
def song_type(df1,df2):
    for i in range(len(df2)):
        if df1['uri'].values[0] == df2['uri'].values[i]:
            df1['song_type']=df2['song_type'].values[i]
            break
        elif df1['uri'].values[0]!=df2['uri'].values[i]:
             df1['song_type']="nothot"
    return df1
    
    
def none(df1,df2):
    cprint('************************************************************','green',attrs=['bold'])
    print()
    if df1['uri'][0]==None:
        cprint("We couldn't find your Song. Let's see how you would feel about this one ;) ",'yellow',attrs=['bold'])
        print()
        df1=df2.sample(1)
        df1=df1[['title','artist','uri','song_type']]
    else:
        cprint("Coool, then you gonna love this one too: ",'cyan',attrs=['bold'])
        print()
        df1['uri']=df1['uri']
        df1=df1[['title','artist','uri','song_type']]
    return df1
    
def get_featurs(df):
    output = pd.DataFrame()
    my_dict = sp.audio_features(df.loc[0,['uri']])[0]
    my_dict_new = {key : my_dict[key] for key in list(my_dict.keys()) }
    my_dict_new['title'] = df['title'][0]
    my_dict_new['artist'] = df['artist'][0]
    output = output.append(my_dict_new, ignore_index=True) 
    return output
    
    
def load(filename): 
    try: 
        with open(filename, "rb") as file: 
            return pickle.load(file) 
    except FileNotFoundError: 
        print("File not found!") 
        
def recommend(title, artist, df,df1,clusters):
    song_type=df['song_type'].values[0]
    df_cluster = df1[df1['Cluster_kmean'] == int(clusters)]
    df_type = df_cluster[df_cluster['song_type'] == str(song_type)]
    sample = df_type.sample()
    while (sample['title'].values[0] == title) and (sample['artist'].values[0] == artist):
        df_type.drop(sample.index)
        if df_type.empty:
            df_type = df_cluster[df_cluster['song_type'] == 'nothot']
        sample = df_type.sample()
    return sample[['artist','title']]

def get_url(sp, title, artist_name):
    results = sp.search(q=title, limit=5)
    for i in range(len(results['tracks']['items'])):
        ar=results['tracks']['items'][i]['album']['artists'][0]['name']
        if (ar in artist_name) or (artist_name in ar):
            return results['tracks']['items'][i]['album']['artists'][0]['external_urls']['spotify']
    return None
    
def final_df(df,url):
    df['url']=url
    print('\033[1m',df.iloc[0,1] ,'\033[0m',"  by  " ,'\033[1m' ,df.iloc[0,0],'\033[0m')
    print("You can access it from here: " + df.iloc[0,2])
    return None

def recommndation():
    cprint("M^2 Production",'grey', attrs=['bold','reverse', 'blink'])
    print("    ")
    cprint('************************************************************','green',attrs=['bold'])
    title = input ("What is your favorite song? ")
    artist = input ("Who is the Artist ? ")
    title=title.title()
    artist=artist.title()
    data_base=pd.read_csv('/Users/mojgun/Documents/IRONHACK/Lab/WEEK6/Day3/Clustering-the-songs-from-the-databases/song_cluster.csv')
    data = {'title':  [title],
        'artist': [artist],
         'uri':[get_uri(sp,title,artist)]
        }
    df = pd.DataFrame(data)
    df1=song_type(df,data_base)
    df1=none(df,data_base)
    df1=df1[['title','artist','uri','song_type']]
    df1=df1.reset_index(drop=True)
    df=get_featurs(df1)
    X=df[["energy",  "speechiness", 
          "acousticness", "instrumentalness",
          "loudness","tempo","danceability",'valence',
          "liveness", "time_signature", "key"]]
    scaler = load("scaler.pickle")
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns = X.columns)
    path="/Users/mojgun/Documents/IRONHACK/Lab/WEEK6/Day3/Clustering-the-songs-from-the-databases/kmeans_11.pickle"
    kmeans=load(path)
    clusters = kmeans.predict(X_scaled_df)
    df1['Cluster_kmean']=clusters
    output=recommend(title, artist, df1,data_base,clusters)
    url=get_url(sp,output['title'].values[0],output['artist'].values[0])
    return final_df(output,url)

  