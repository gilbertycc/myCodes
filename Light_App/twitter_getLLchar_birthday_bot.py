# import required modules
from SPARQLWrapper import SPARQLWrapper2
from datetime import datetime
import pytz
import tweepy
import time


# overide the old MyOAuth2UserHandler clasee for token refresh purpose
class MyOAuth2UserHandler(tweepy.OAuth2UserHandler):
    # Kudos https://github.com/tweepy/tweepy/pull/1806

    def refresh_token(self, refresh_token): 
        new_token = super().refresh_token(
                "https://api.twitter.com/2/oauth2/token",
                refresh_token=refresh_token,
                body=f"grant_type=refresh_token&client_id={self.client_id}",
            )
        return new_token

def loggingOutput(logText):
    logPath = './log/output.log'

    currentTimestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if isinstance(logText, str):
        formatString = '[' + currentTimestamp + ']' + logText + '\n'

        with open(logPath, 'a') as logFile:
            logFile.write(formatString)
    else:
        with open(logPath, 'a') as logFile:
            logFile.write(logText)
            logFile.write('\n')


def createTweet(accessToken, tweetText):
    # initialize client with consumer and access tokens
    client = tweepy.Client(accessToken)

    # create tweet with timestamp = time.time()
    # timestamp = str(time.time())
    
    #response = client.create_tweet(text="This Tweet was Tweeted using Tweepy and Twitter API v2! " + timestamp,user_auth=False)
    response = client.create_tweet(text=tweetText, user_auth=False)
    loggingOutput("https://twitter.com/user/status/" + response.data['id'])
    
    # return the tweet id after created the tweet
    return (response.data['id'])
    
def deleteTweet(accessToken, tweetID):
    # initialize client with consumer and access tokens
    client = tweepy.Client(accessToken)
    response = client.delete_tweet(tweetID, user_auth=False)

    loggingOutput(response)



def getTwitterAccessToken(client_id,client_secret,redirect_uri):

    authorization_response = ""
    inputFilePath = './log/respondString.txt'
    
    # step 1 authenticate app
    oauth2_user_handler = tweepy.OAuth2UserHandler(client_id=client_id,
                                                   redirect_uri=redirect_uri,
                                                   scope=['tweet.read', 'tweet.write', 'users.read','offline.access'],
                                                   client_secret=client_secret)

    # print(oauth2_user_handler.get_authorization_url())
    loggingOutput(oauth2_user_handler.get_authorization_url())

    # authorization_response = input('Paste redirect url here:  ')

    with open(inputFilePath, "w") as respondStringFile:
        respondStringFile.truncate()

    
    while (True):
        with open(inputFilePath) as respondStringFile:
            authorization_response = respondStringFile.readline()
        
        if (len(authorization_response) > 0):
            break
            
        loggingOutput(inputFilePath + " is still empty, waiting for the input...")
        time.sleep(5)

    loggingOutput(authorization_response)

    # fetch access token
    access_token = oauth2_user_handler.fetch_token(authorization_response)
    return (access_token)

def refreshTwitterAccessToken(client_id,client_secret,redirect_uri,accToken):
    auth = MyOAuth2UserHandler(
                                 client_id=client_id,  # same credentials as used before
                                 client_secret= client_secret,
                                 redirect_uri=redirect_uri,
                                 scope=['tweet.read', 'tweet.write', 'users.read','offline.access'],
                                 )

    newToken = auth.refresh_token(accToken["refresh_token"])
    return (newToken)


def getCDateAndFormat(fString):
    jpTimeZone = pytz.timezone("Asia/Tokyo") 
    currentDateAndTime = datetime.now(jpTimeZone)
    return (currentDateAndTime.strftime(fString))


def getTodayBDayLilies():

    # Usage:
    # 1. Pre-requsis: install SPARQLWrapper from PyPI 
    #  pip install sparqlwrapper
    # 2. API Help: https://github.com/Assault-Lily/LuciaDB
    
    # define variables 
    BaseURL = "https://luciadb.assaultlily.com/sparql/query"
    cDate = getCDateAndFormat("--%m-%d")
    
    sparql = SPARQLWrapper2(BaseURL)
    sparql.setQuery("""
        PREFIX schema: <http://schema.org/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX lily: <https://luciadb.assaultlily.com/rdf/IRIs/lily_schema.ttl#>

        SELECT ?name ?brithDate
        WHERE {
          ?lily rdf:type lily:Lily;
                lily:nameKana ?namekana;
                schema:birthDate ?brithDate.
          #FILTER(CONTAINS(?namekana,"あ"))
          ?lily schema:name ?name.
          FILTER(lang(?name)="ja")
          
        }
        """
                    )

    # Print brithdate lilies in current month
    #for result in sparql.query().bindings:
    #    bDate = str(result['brithDate'])[15:-2]
    #    tdate = getCDateAndFormat("--%m")
    #    if tdate in bDate:
    #        print (str(result['name'])[15:-2],bDate)
    
    # Print brithdate lilies today
    #print (cDate)
    # Enable below for debug only
    # cDate = "--12-25"
    # print ("今日は"+cDate[2:4]+"月"+cDate[5:7]+"日。。。")
    
    cbLilies=[]
    cbLilies.append(cDate[2:4]+"月"+cDate[5:7]+"日")
    
    for result in sparql.query().bindings:
        bDate = str(result['brithDate'])[15:-2]
        
        if cDate in bDate:
            #print (str(result['name'])[15:-2],bDate)
            cbLilies.append(str(result['name'])[15:-2])
    
    return (cbLilies)





def main():

    # define variables
    # define twitter app credentials
    twitter_client_id = '<set id>'
    twitter_client_secret = '<set secret>'
    twitter_redirect_uri = 'https://127.0.0.1'

    loopSleepSec = 3600
    resetDate = 0
    currentDate = getCDateAndFormat("--%m-%d")
    cDateString = (currentDate[2:4]+"月"+currentDate[5:7]+"日")

    # Starting Twitter API OAuth2 authentication flow & retrieve the access_token for the 1st time
    access_token = getTwitterAccessToken(twitter_client_id,twitter_client_secret,twitter_redirect_uri)



    while(1):

        # refresh the client token 
        access_token = refreshTwitterAccessToken(twitter_client_id,twitter_client_secret,twitter_redirect_uri,access_token)
        
        # check and return the list of today Brithday lilies
        bdLilyList = getTodayBDayLilies()

        if len(bdLilyList) == 1 and resetDate == 0:
            loggingOutput("今日が誕生日のリリィはありません("+bdLilyList[0]+")。")
        elif len(bdLilyList) > 1 and resetDate == 0:
            twTextHead = "本日("+bdLilyList[0]+")は"
            twTextTail = "のお誕生日です、誕生日おめでとう！"+"\n\n　#アサルトリリィ　\n　#ラスバレ　\n　#アサルトリリィ誕生日祝"
            twTextMid = ""
            
            for i in range(1, len(bdLilyList)):
                twTextMid = twTextMid + bdLilyList[i] + " "
                twTextTail = twTextTail + "\n　#" + bdLilyList[i] + "生誕祭" + datetime.now().strftime('%Y')
            
            twText = twTextHead + twTextMid + twTextTail
            createTweet(access_token["access_token"], twText)
            resetDate = 1
        elif (resetDate == 1 and bdLilyList[0] != cDateString):
            resetDate = 0
 
        currentDate = getCDateAndFormat("--%m-%d")
        cDateString = (currentDate[2:4]+"月"+currentDate[5:7]+"日")
        
        

        #for Debug
        #deleteTweet(access_token["access_token"], tweetID)
        # loggingOutput(access_token)
        # loggingOutput(str("[Debug] "+cDateString,bdLilyList[0]+resetDate))
        loggingOutput('[Debug] *** Alive heartbeat ***')
        time.sleep(loopSleepSec)
    

##### MAIN #####
if __name__ == "__main__":
    main()