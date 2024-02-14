import wolframalpha
from pprint import pprint
import requests
import os
import urllib.parse

print("Extracting information using Full results API, First Service")
appid = 'xxxxxxxxxxxxx'

query = urllib.parse.quote_plus("What is an engine")

query_url = f"http://api.wolframalpha.com/v2/query?" \
             f"appid={appid}" \
             f"&input={query}" \
             f"&format=plaintext" \
             f"&output=json"
r = requests.get(query_url).json()

data = r["queryresult"]['pods'][1]['subpods'][0]
text = data["plaintext"]

print('Question: What is an engine, answer is\n', text)

query2 = urllib.parse.quote_plus("")

print()
print('Solving a math equation using Full results API, Second Service')

equation = "4 + 2x = 10"
query = urllib.parse.quote_plus(f"solve {equation}")
query_url = f"http://api.wolframalpha.com/v2/query?" \
            f"appid={appid}" \
            f"&input={query}" \
            f"&includepodid=Result" \
            f"&output=json"

r = requests.get(query_url).json()

data = r["queryresult"]["pods"][0]["subpods"][0]
plaintext = data["plaintext"]

print('The solution to the equation is this', plaintext)

print()
print('using conversational API, asking questions, First Service')

question = "Who is the president of the united states "
query_url = f"http://api.wolframalpha.com/v1/conversation.jsp?" \
            f"appid={appid}" \
            f"&geolocation={appid}" \
            f"&i={question}" \

r = requests.get(query_url).json()
answer = r["result"]
conversation_id = r["conversationID"]
host = r["host"]

print('The answer to \'Who is the president of the united states\' is ', answer)

followup_question = "what is the start date"
query_url = f"http://{host}/api/v1/conversation.jsp?" \
            f"appid={appid}" \
            f"&conversationID={conversation_id}" \
            f"&i={followup_question}" \

r = requests.get(query_url).json()
answer2 = r['result']
print('The answer to \'what is the start date\' is ', answer2)

print()
print('Using Conversational API, second Service')

question = "Who was Neil Armstrong"
query_url = f"http://api.wolframalpha.com/v1/conversation.jsp?" \
            f"appid={appid}" \
            f"&geolocation={appid}" \
            f"&i={question}" \

r = requests.get(query_url).json()
answer = r['result']
conversation_id = r["conversationID"]
host = r["host"]
print('The answer to \' Who was Neil Armstrong is\' ', answer)


followup_question = "Place of birth"
query_url = f"http://{host}/api/v1/conversation.jsp?" \
            f"appid={appid}" \
            f"&conversationID={conversation_id}" \
            f"&i={followup_question}" \

r = requests.get(query_url).json()
answer2 = r['result']
print('The answer to \'Place of birth\' is ', answer2)


