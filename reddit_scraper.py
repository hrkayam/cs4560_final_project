from ntlk_reddit_scraper import RedditScraper
# import praw
# import string

rs = RedditScraper()
rs.scrape()

#
# def parse(body):
#     body = str(body)
#     # print(body)
#     sentence_list = []
#     tokenized_sentence_list = []
#     table = str.maketrans('', '', string.punctuation)
#     sentence = ""
#     for char in body:
#         if char is not ".":
#             sentence += char
#         else:
#             sentence.translate(table)
#             sentence.replace("&nbsp;", "")
#             sentence.replace("\n", "")
#             sentence_list.append(sentence.strip())
#             tokenized_sentence_list.append(sentence.split())
#             sentence = ""
#     return sentence_list, tokenized_sentence_list
#
# reddit = praw.Reddit(client_id='WVIjShUtj9WN4Q', client_secret='QHArhM9JsfcnFC3wj6cUr4ImshA', user_agent='CS 4650 NLP Project Kayam Tamhankar')
#
# neutral_posts = reddit.subreddit('news').hot(limit=1)
# f = open("corpus.txt", "w")
#
# sentence_list, tokenized_sentence_list = parse("Neha Kakkar (born 6 June 1988) is an Indian playback singer. She began performing at religious events at the age of four and participated in the second season of the singing reality show, Indian Idol, in which she was eliminated early in the show. After several struggles in her career, she made her Bollywood debut as a chorus singer in the film Meerabai Not Out (2008). She rose to prominence upon the release of the dance track \"Second Hand Jawani\" from Cocktail (2012), which was followed by several other popular party songs including \"Sunny Sunny\" from Yaariyan (2014) and \"London Thumakda\" from Queen (2014).")
# for i in range(len(sentence_list)):
#     print("------------")
#     print(sentence_list[i])
#     f.write(sentence_list[i] + "\n")
# print(len(sentence_list))
#
# # for post in neutral_posts:
#     # sentence_list = parse(post.selftext)
#     # for sentence in sentence_list:
#     #     print("------------")
#     #     print(sentence)
#     #     f.write(sentence + "\n")
#     # print(len(sentence_list))
# f.close()
