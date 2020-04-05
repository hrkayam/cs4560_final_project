import nltk
import praw
import string



class RedditScraper(object):

    def __init__(self, subreddit = "news", length_cap = 100, hot_cap = 10, filename="corpus.txt"):
        self.subreddit = subreddit
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
                if word is "\'":
                    words[-1] = words[-1] + word
            # words = [word.lower() for word in tokenized_sentence if word.isalpha()]
            tokenized_sentence_list.append(words)
        return sentence_list, tokenized_sentence_list

    def scrape(self):
        reddit = praw.Reddit(client_id='WVIjShUtj9WN4Q', client_secret='QHArhM9JsfcnFC3wj6cUr4ImshA', user_agent='CS 4650 NLP Project Kayam Tamhankar')
        neutral_posts = reddit.subreddit(self.subreddit).hot(limit=self.hot_cap)
        f = open(self.filename, "w")
        for post in neutral_posts:
            sentence_list, tokenized_sentence_list = self.parse(post.selftext)

            for i in range(len(sentence_list)):
                for word in tokenized_sentence_list[i]:
                    f.write(word + " ")
                f.write("\n")
        f.close()
