import numpy as np
import copy


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

total_len = file_len('small_corpus_tagged.txt')

user_dict = {}

usercomments_dict = {}

filepath = 'small_corpus_tagged.txt'
with open(filepath) as fp:
   line = fp.readline()
   indexer = -1
   while line:
       ind = line.find(']')
       user = line[1:ind]

       if user not in user_dict:
           indexer += 1
           user_dict[user] = indexer
           usercomments_dict[user] = ''
       line = fp.readline()

#print(user_dict)

one_hot_vecs = np.zeros((total_len, len(user_dict)))

print(usercomments_dict)

with open(filepath) as fp:
   line = fp.readline()
   ind = 0
   while line:
       userInd = line.find(']')
       user = line[1:userInd]

       userComment = line[userInd + 1:-1]

       arrInd = user_dict[user]
       one_hot_vecs[ind][arrInd] = 1
       usercomments_dict[user] = usercomments_dict[user] + userComment
       ind += 1
       line = fp.readline()

print(one_hot_vecs)
print(usercomments_dict)
