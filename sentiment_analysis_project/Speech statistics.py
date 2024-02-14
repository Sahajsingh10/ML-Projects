import re
import readability
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


print("1. Donald J Trump\n2. John F Kennedy")
a = int(input("Which speech do you want to Analyze\n Enter number "))
if a == 1:
    t = "Donald J Trump"
elif a == 2:
    t = "John F Kennedy"

with open(t) as outFile:
    data = outFile.read()
    for i in data:
        if i in '!,>,<,?\',.,-,;,:,\",\"':
            data = data.replace(i, "")
    for i in data:
        if i in '\n':
            data = data.replace(i, " ")

    sentence = data.split('\n')
    words = data.split(" ")
    data3 = re.findall("[a-zA-Z0-9]", data)
    print("The word count for this speech is ", len(words))
    print("No. of characters are", len(data3))
    len_words = 0
    for i in words:
       len_words = len_words + (len(i))

    len_sentence = 0
    for i in sentence:
        len_sentence = len_sentence + (len(i))

    print("The average word length is", len_words /len(words))
    print("The average sentence length is", len_sentence / len(sentence))

    d = dict()
    e = dict()
    count = 0
    for i in words:
        if i in d:
            d[i] = d[i] + 1
        else:
            d[i] = 1
    print()

    print('Frequencies of the first 15 words in ascending order(by values):')
    for key,val in d.items():
        count = count + 1
        if count == 17:
            break
        e[key] = val

    sort_orders = sorted(e.items(), key=lambda x: x[1])
    for i in sort_orders:
        print(i[0],i[1])

    print()
    largestWord = ""
    print('The 10 largest word in descending order: ')
    for i in words:
        if len(largestWord) < len(i):
            largestWord = i
    print(largestWord)

    for i in words:
        if len(i) == 17:
            sec_large = i
    for i in words:
        if len(i) == 15:
            thir_large = i
    print(thir_large)
    for i in words:
        if len(i) == 14:
            forth_large = i
    print(forth_large)
    for i in words:
        if len(i) == 13:
            fifth_large = i
    print(fifth_large)
    for i in words:
        if len(i) == 12:
            six_large = i
    print(six_large)
    for i in words:
        if len(i) == 11:
            seven_large = i
    print(seven_large)
    for i in words:
        if len(i) == 10:
            eight_large = i
    print(eight_large)
    for i in words:
        if len(i) == 9:
            ninth_large = i
    print(ninth_large)
    for i in words:
        if len(i) == 8:
            tenth_large = i
    print(tenth_large)
    print()
    results = readability.getmeasures(data, lang='en')
    print("The readability measure for this speech is", results['readability grades']['FleschReadingEase'])


    wordcloud = WordCloud(width=2000, height=1000, random_state=1, background_color='black', colormap='Pastel1',
      collocations=False, stopwords=STOPWORDS).generate(data)

    plt.figure(figsize=(40, 30))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


