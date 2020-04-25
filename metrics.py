import math

#PERPLEXITY - UNIGRAM

word_counts = {}
word_prob = {}
total_words = 0

with open('small_corpus_tagged.txt','r') as file:
    for line in file:
        for word in line.split():
            total_words += 1
            if word in word_counts.keys():
                word_counts[word] += 1
            else:
                word_counts[word] = 1
for word in word_counts:
    word_prob[word] = word_counts[word] / total_words

with open('small_corpus_tagged.txt','r') as file:
    exponent = 0
    for line in file:
        sentence_prob = 1
        for word in line.split():
            sentence_prob += math.log(word_prob[word], 2)
        exponent += sentence_prob
        sentence_prob = 1

exponent /= total_words
perplexity = 2**(-exponent)

print(len(word_counts.keys()))
print(total_words)
print(perplexity)
