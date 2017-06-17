import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network
from sklearn.metrics import average_precision_score

FLAGS = tf.app.flags.FLAGS
#change the name to who you want to send
#tf.app.flags.DEFINE_string('wechat_name', 'Tang-24-0325','the user you want to send info to')
# tf.app.flags.DEFINE_string('wechat_name', 'filehelper','the user you want to send info to')




#embedding the position
def pos_embed(x):
	if x < -60:
		return 0
	if x >= -60 and x <= 60:
		return x+61
	if x > 60:
		return 122

#find the index of x in y, if x not in y, return -1
def find_index(x,y):
	flag = -1
	for i in range(len(y)):
		if x != y[i]:
			continue
		else:
			return i
	return flag


	
print 'reading word embedding data...'
vec = []
word2id = {}
f = open('/Users/samitha/Documents/Project/new_may01/TensorFlow-NRE-master/origin_data/vec.txt')
f.readline()
while True:
	content = f.readline()
	if content == '':
		break
	content = content.strip().split()
	word2id[content[0]] = len(word2id) ## word list in dic 1,2, 3 ..
	content = content[1:]
	content = [(float)(i) for i in content]
	vec.append(content)
f.close()
word2id['UNK'] = len(word2id)
word2id['BLANK'] = len(word2id)

dim = 50
vec.append(np.random.normal(size=dim,loc=0,scale=0.05))
vec.append(np.random.normal(size=dim,loc=0,scale=0.05))
vec = np.array(vec,dtype=np.float32)


print 'reading relation to id'
relation2id = {}	
f = open('/Users/samitha/Documents/Project/new_may01/TensorFlow-NRE-master/origin_data/relation2id.txt','r')
while True:
	content = f.readline()
	if content == '':
		break
	content = content.strip().split()
	relation2id[content[0]] = int(content[1])
f.close()

#length of sentence is 70
fixlen = 70
#max length of position embedding is 60 (-60~+60)
maxlen = 60

train_sen = {} #{entity pair:[[[label1-sentence 1],[label1-sentence 2]...],[[label2-sentence 1],[label2-sentence 2]...]}
train_ans = {} #{entity pair:[label1,label2,...]} the label is one-hot vector





print('reading test data ...')

test_sen = {} #{entity pair:[[sentence 1],[sentence 2]...]}
test_ans = {} #{entity pair:[labels,...]} the labels is N-hot vector (N is the number of multi-label)

f = open('/Users/samitha/Documents/Project/new_may01/TensorFlow-NRE-master/origin_data/my_test.txt','r')

while True:
	content = f.readline()
	if content == '':
		break
	
	content = content.strip().split()
	en1 = content[2]
	en2 = content[3]
	#positions of 2 entities   imp: entities must be sanke_cased fixed from stanford nlp OPEN ER
	relation = 0
	if content[4] not in relation2id:
		relation = relation2id['NA']
	else:
		relation = relation2id[content[4]]
	 		
	tup = (en1,en2)    #('aaa', 'bbb')
	
	if tup not in test_sen:
		#put the same entity pair sentences into a dict 
		#{('aaa', 'bbb'): [[11, 12, 13, 0.0, 0.0], [21, 22, 23, 0.0, 0.0]]}
		test_sen[tup]=[]
		y_id = relation
		label_tag = 0
		label = [0 for i in range(len(relation2id))]
		label[y_id] = 1
		test_ans[tup] = label
	else:
		y_id = relation
		test_ans[tup][y_id] = 1
		
	sentence = content[5:-1]

	en1pos = 0
	en2pos = 0
	
	for i in range(len(sentence)):
		if sentence[i] == en1:
			en1pos = i
		if sentence[i] == en2:
			en2pos = i
	output = []

	for i in range(fixlen):
		word = word2id['BLANK']
		rel_e1 = pos_embed(i - en1pos)
		rel_e2 = pos_embed(i - en2pos)
		output.append([word,rel_e1,rel_e2])

	for i in range(min(fixlen,len(sentence))):
		word = 0
		if sentence[i] not in word2id:
			word = word2id['UNK']
		else:
			word = word2id[sentence[i]]

		output[i][0] = word
	test_sen[tup].append(output)

print(test_sen)

	#numpy arrys without tuple id's [[11, 12, 13, 0.0, 0.0], [21, 22, 23, 0.0, 0.0]]

train_x = []
train_y = []
test_x = []
test_y = []
print(train_x)

print 'organizing train data'
f = open('/Users/samitha/Documents/Project/new_may01/TensorFlow-NRE-master/data/my_train_q&a.txt','w')
temp = 0
for i in train_sen:
	if len(train_ans[i]) != len(train_sen[i]):
		print 'ERROR'
	lenth = len(train_ans[i])
	for j in range(lenth):
		train_x.append(train_sen[i][j])
		train_y.append(train_ans[i][j])
		f.write(str(temp)+'\t'+i[0]+'\t'+i[1]+'\t'+str(np.argmax(train_ans[i][j]))+'\n')
		temp+=1
f.close()

print 'organizing test data'
f = open('/Users/samitha/Documents/Project/new_may01/TensorFlow-NRE-master/data/my_test_q&a.txt','w')
temp=0
for i in test_sen:		
	test_x.append(test_sen[i])
	test_y.append(test_ans[i])
	tempstr = ''
	for j in range(len(test_ans[i])):
		if test_ans[i][j]!=0:
			tempstr = tempstr+str(j)+'\t'
	f.write(str(temp)+'\t'+i[0]+'\t'+i[1]+'\t'+tempstr+'\n')
	temp+=1
f.close()

test_x = np.array(test_x)
test_y = np.array(test_y)

pall_test_x = []
pall_test_y = []
for i in range(len(test_x)):
	print("len test_x[i]"+str(len(test_x[i])))
	if len(test_x[i]) > 0:
		
		pall_test_x.append(test_x[i])
		pall_test_y.append(test_y[i])


pall_test_x = np.array(pall_test_x)
pall_test_y = np.array(pall_test_y)


print 'reading p-all test data'
x_test = pall_test_x
print 'seperating p-all test data'
test_word = []
test_pos1 = []
test_pos2 = []

for i in range(len(x_test)):
	word = []
	pos1 = []
	pos2 = []
	for j in x_test[i]:
		temp_word = []
		temp_pos1 = []
		temp_pos2 = []
		for k in j:
			temp_word.append(k[0])
			temp_pos1.append(k[1])
			temp_pos2.append(k[2])
		word.append(temp_word)
		pos1.append(temp_pos1)
		pos2.append(temp_pos2)
	test_word.append(word)
	test_pos1.append(pos1)
	test_pos2.append(pos2)

test_word = np.array(test_word)
test_pos1 = np.array(test_pos1)
test_pos2 = np.array(test_pos2)

np.save('/Users/samitha/Documents/Project/new_may01/TensorFlow-NRE-master/data/gpall_test_word.npy',test_word)
np.save('/Users/samitha/Documents/Project/new_may01/TensorFlow-NRE-master/data/gpall_test_pos1.npy',test_pos1)
np.save('/Users/samitha/Documents/Project/new_may01/TensorFlow-NRE-master/data/gpall_test_pos2.npy',test_pos2)
np.save('/Users/samitha/Documents/Project/new_may01/TensorFlow-NRE-master/data/gpall_test_y.npy',pall_test_y)

