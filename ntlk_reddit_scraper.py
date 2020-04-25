import nltk
import praw
import string

class RedditScraper(object):

    def __init__(self, subreddits = ["askreddit"], length_cap = 100, hot_cap = 1000, filename="corpus_tagged.txt"):
        self.subreddits = subreddits
        self.length_cap = length_cap
        self.hot_cap = hot_cap
        self.filename = filename

    def parse(self, body):
        body = str(body)
        # print(body)
        sentence_list = nltk.sent_tokenize(body)
        tokenized_sentence_list = []
        for sentence in sentence_list:
            tokenized_sentence = nltk.word_tokenize(sentence)
            words = []
            for word in tokenized_sentence:
                if word.isalpha():
                    if len(words) > 0 and words[-1] is "\'":
                        words[-1] = words[-1] + word
                    else:
                        words.append(word)
                if len(words) > 0 and word is "\'":
                    words[-1] = words[-1] + word
            # words = [word.lower() for word in tokenized_sentence if word.isalpha()]
            tokenized_sentence_list.append(words)
        return sentence_list, tokenized_sentence_list

    def scrape(self):
        reddit = praw.Reddit(client_id='WVIjShUtj9WN4Q', client_secret='QHArhM9JsfcnFC3wj6cUr4ImshA', user_agent='CS 4650 NLP Project Kayam Tamhankar')
        f = open(self.filename, "w")
        for sr in self.subreddits:
            print(sr)
            count = 0
            for comment in reddit.subreddit(sr).stream.comments():
                if count > 10000:
                    break
                if count % 200 == 0:
                    print(count)
                user = comment.author.name
                sentence_list, tokenized_sentence_list = self.parse(comment.body)
                count += 1
                for i in range(len(sentence_list)):
                    f.write('[' + user + '] ')
                    for word in tokenized_sentence_list[i]:
                        try:
                            f.write(word + " ")
                        except UnicodeEncodeError:
                            f.write("[UNKNOWN]" + " ")
                    f.write("\n")
        f.close()

    def scrape_user_comments(self, username):
        reddit = praw.Reddit(client_id='WVIjShUtj9WN4Q', client_secret='QHArhM9JsfcnFC3wj6cUr4ImshA', user_agent='CS 4650 NLP Project Kayam Tamhankar')

        user = reddit.redditor(username)

        f = open("user_" + username, "w")
        f.write(username + "\n")
        for sr in self.subreddits:
            print(sr)
            count = 0
            for comment in user.comments.new(limit=None):
                if count > 10000:
                    break
                if count % 200 == 0:
                    print(count)
                sentence_list, tokenized_sentence_list = self.parse(comment.body)
                count += 1
                for i in range(len(sentence_list)):
                    for word in tokenized_sentence_list[i]:
                        try:
                            f.write(word + " ")
                        except UnicodeEncodeError:
                            f.write("[UNKNOWN]" + " ")
                    f.write("\n")
        f.close()
