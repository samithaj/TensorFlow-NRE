import requests
import json
import sys
from nltk.stem.snowball import SnowballStemmer
from owl_generator import *

import os ,sys
PACKAGE_PARENT2 = 'nre'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT2)))


import my_test_GRU as nre
endpoint = "http://192.168.8.102:9000"
def indentify_pos(data):

	online_endpoint = "http://corenlp.run"
	print("------start---")
	print (data)

	# if len(sys.argv) == 1:
	# 	# data = "400 victims in Dnkoluwaththa in Pitabaddra . Not enough food to eat. Very hard situation there"
	# else:
	# 	data = sys.argv[1]
	properties = { 'annotators': 'ner, depparse,openie', 'outputFormat': 'json' }
	response = requests.post(endpoint, params = { 'properties': str(properties) }, data = data.encode(), headers = {'Connection': 'close'})
	output = json.loads(response.text, encoding = 'utf-8', strict = True)
	# print(json.dumps(output, indent=4))

	sentence = output['sentences'][0]
	dependency_dict = {}
	triple_dict = {}
	pos_dict = {}

	# print(json.dumps(sentence, indent=4))

	for dependency in sentence['basicDependencies']:
		if dependency["dep"] == 'compound':
			dependency_dict[dependency["dependentGloss"]] = dependency["dep"]


	for dependency in sentence['basicDependencies']:
		dependency_dict[dependency["dep"]] = dependency["dependentGloss"]


	triples = sentence['openie']
	print("--------")
	# print(dependency_dict)
	print("--------")

	# print(json.dumps(sentence['openie'], indent=4))






	# dependency_dict["ROOT"] = SnowballStemmer("english").stem(dependency_dict["ROOT"])

	# print("-----dependency_dict--------")
	# for key in dependency_dict:
	# 	# print("%s: %s" % (key, dependency_dict[key]))

	ner_dict = {}


	words=[]
	ner=[]
	for token in sentence['tokens']:

			print("%s is a %s" % (token["word"], token["ner"]))
			ner_dict[token["word"]] =  token["ner"]
			tmp1= token["word"]
			tmp2=token["ner"]
			words.append(tmp1)
			ner.append(tmp2)

	new_ner_dict ={}
	toRemove =[]
	for idx, word in enumerate(words):

		nerNow = ner[idx]
		tmp =0
		if nerNow !=  "O":
			if (idx+1)<len(words):
				if (ner[idx+1] == nerNow):
					tmp =1
					toRemove.append(word)
					toRemove.append(words[idx+1])
					new_ner_dict[word+"_"+words[idx+1]] = ner[idx]

			if tmp != 1:
				new_ner_dict[word] = ner[idx]

	for key in toRemove:
		new_ner_dict.pop(key, None)


	entities=[]
	print("-----tokens-ner--------")
	for key in new_ner_dict:
		print("%s: %s" % (key, new_ner_dict[key]))
		entities.append(key)



	print("-----tokens-pos--------")
	for token in sentence['tokens']:
		if token["pos"] != "O":
			# print("%s pos-> %s" % (token["word"], token["pos"]))
			pos_dict[token["word"]] =  token["pos"]

	print("---triple_dict-----")
	print(triples)
	print("--------")

	# for triple in sentence['openie']:
	# 	if pos_dict[triple["subject"]]=='NN' or pos_dict[triple["subject"]]=='NNP':
	# 		print("subject is a entitiy (instance of a class) ")
	# 		if ner_dict[triple["subject"]] == 'PERSON':
	# 			print(triple["subject"] +" is a PERSON")
    #
	# 	if pos_dict[triple["object"].split(" ")[0]]=='NN' or pos_dict[triple["object"].split(" ")[0]]=='NNP':
	# 		print("object is a entitiy (instance of a class) ")
	# 		if ner_dict[triple["object"].split(" ")[0]] == 'LOCATION':
	# 			print(triple["object"] +" is a LOCATION")

	print("---ner_dict-----")
	status_code = 0
	if len(new_ner_dict)>1:

		print("Have more than one entities,possible combinations: ")

		entityPairs=[]
		entityPairs =list(choose_iter(entities, 2))
		print(entityPairs)

		# if 'PERSON' in new_ner_dict.values():
        #
		# 	print ("have a person")
		# 	for entity in new_ner_dict:
		# 		print("%s: %s" % (entity, new_ner_dict[entity]))
		# 		if new_ner_dict[entity]=='PERSON':
		# 			one=entity
		# 		two = entity
		# else:
		# 	print ("dont have a person")
        #
		# 	for entity in new_ner_dict:
		# 		print("%s: %s" % (entity, new_ner_dict[entity]))
        #
		# 		entityPair.append(entity)





		results =[]
		validResults=[]



		for entityPair in entityPairs:

			data = data.replace(entityPair[0].replace("_"," "), entityPair[0].lower())
			data = data.replace(entityPair[1].replace("_", " "), entityPair[1].lower())

			print(entityPair[0], entityPair[1], "/location/location/contains", data)
			finalOutput = nre.test(entityPair[0], entityPair[1], "/location/location/contains", data)
			print("-----output------")
			print (finalOutput)

			if finalOutput!='NA':
				validResults.append((entityPair[0], entityPair[1], finalOutput))

			results.append((entityPair[0],entityPair[1],finalOutput))

			print(entityPair[1], entityPair[0], "/location/location/contains", data)
			finalOutput = nre.test(entityPair[1], entityPair[0], "/location/location/contains", data)
			print("-----output------")
			print (finalOutput)
			if finalOutput!='NA':
				validResults.append((entityPair[1], entityPair[0], finalOutput))

			results.append((entityPair[1], entityPair[0], finalOutput))


# -----------lower case
			print(entityPair[0].lower(), entityPair[1].lower(), "/location/location/contains", data)
			finalOutput = nre.test(entityPair[0].lower(), entityPair[1].lower(), "/location/location/contains", data)
			print("-----output------")
			print (finalOutput)

			if finalOutput!='NA':
				validResults.append((entityPair[0].lower(), entityPair[1].lower(), finalOutput))

			results.append((entityPair[0].lower(),entityPair[1].lower(),finalOutput))

			print(entityPair[1].lower(), entityPair[0].lower(), "/location/location/contains", data)
			finalOutput = nre.test(entityPair[1].lower(), entityPair[0].lower(), "/location/location/contains", data)
			print("-----output------")
			print (finalOutput)
			if finalOutput!='NA':
				validResults.append((entityPair[1].lower(), entityPair[0].lower(), finalOutput))

			results.append((entityPair[1].lower(), entityPair[0].lower(), finalOutput))

			if len(validResults)>0:
				ontology_iri="http://www.semanticweb.org/lily/ontologies/2016/10/disaster-management-system"
				generator = OwlGenerator(ontology_iri)
				entity1 = OwlEntity(validResults[0][0])
				entity2 = OwlEntity(validResults[0][1])
				attr = OwlObjAttr(validResults[0][2], entity1, entity2)
				generator.entities.append(entity1)
				generator.entities.append(entity2)
				generator.obj_attrs.append(attr)
				generator.write_owl("test.owl")

				status_code = 1
			print results


	return status_code

		# print(recognise_ne("http://www.semanticweb.org/lily/ontologies/2016/10/disaster-management-system", "test.owl",
		# 			 'Lilan is a student'))









def choose_iter(elements, length):
	for i in xrange(len(elements)):
		if length == 1:
			yield (elements[i],)
		else:
			for next in choose_iter(elements[i + 1:len(elements)], length - 1):
				yield (elements[i],) + next

def choose(l, k):
	return list(choose_iter(l, k))

def parseTweets(text):

	# endpoint = "http://192.168.8.106:9000"
	online_endpoint = "http://corenlp.run"

	if len(sys.argv) == 1:
		data = text
	else:
		data = sys.argv[1]
	properties = { 'annotators': 'ner, depparse,openie', 'outputFormat': 'json' }
	response = requests.post(endpoint, params = { 'properties': str(properties) }, data = data.encode(), headers = {'Connection': 'close'})
	output = json.loads(response.text, encoding = 'utf-8', strict = True)
	# print(json.dumps(output, indent=4))

	sentence = output['sentences'][0]
	dependency_dict = {}
	triple_dict = {}
	pos_dict = {}
	extractedDetails= {}
	# print(json.dumps(sentence, indent=4))

	for dependency in sentence['basicDependencies']:
		dependency_dict[dependency["dependentGloss"]] = dependency["dep"]

	for dependency in sentence['basicDependencies']:
		dependency_dict[dependency["dep"]] = dependency["dependentGloss"]


	triples = sentence['openie']
	print("--------")
	# print(dependency_dict)
	print("--------")

	# print(json.dumps(sentence['openie'], indent=4))






	dependency_dict["ROOT"] = SnowballStemmer("english").stem(dependency_dict["ROOT"])



	ner_dict = {}
	locations = []
	for eachsentence in output['sentences']:
		for token in eachsentence['tokens']:
			if token["ner"] != "O":

				print("%s is a %s" % (token["word"], token["ner"]))
				ner_dict[token["word"]] =  token["ner"]
				if token["ner"] == "NUMBER":
					if len(token["word"])>9:
						print "length",len(token["word"])
						extractedDetails["phone number"] =token["word"]
				if token["ner"] == "PERSON":
					extractedDetails["contact person"] =token["word"]
				if token["ner"] == "LOCATION":
					locations.append(token["word"])

				if token["ner"] == "DATE":
					extractedDetails["date"] = token["word"]

	extractedDetails["location"] = locations

	for token in sentence['tokens']:
		if token["pos"] != "O":
			# print("%s pos-> %s" % (token["word"], token["pos"]))
			pos_dict[token["word"]] =  token["pos"]

	print("---ner_dict-----")
	# for key in ner_dict:
	# 	# print("%s: %s" % (key, dependency_dict[key]))
	# print("--------")
	items =[]
	for eachsentence in output['sentences']:
		for triple in eachsentence['openie']:
			if triple["relation"]=='need' or triple["relation"]=='needs' or triple["relation"]=='required' or triple["relation"]=='required for':

					print(triple["subject"] +" NEEDS"+triple["object"])
					items.append(triple["object"])
	extractedDetails["wanted items"] =items
	# if pos_dict[triple["object"].split(" ")[0]]=='NN' or pos_dict[triple["object"].split(" ")[0]]=='NNP':
		# 	print("object is a entitiy (instance of a class) ")
		# 	if ner_dict[triple["object"].split(" ")[0]] == 'LOCATION':
		# 		print(triple["object"] +" is a LOCATION")

	print(extractedDetails)

	return extractedDetails

# indentify_pos("400 victims in Dnkoluwaththa in Pitabaddra . Not enough food to eat. Very hard situation there")


	# parseTweets("We need cleaning items, milk packets, clothes for children and underwear for ladies. Please call Daykanthi 071 135 0844. Last Updated: 2017-05-30 2:33pm.")
	# for triple in sentence['openie']:
	# 	if pos_dict[triple["subject"]]=='NN' or pos_dict[triple["subject"]]=='NNP':
	# 		print("subject is a entitiy (instance of a class) ")
	# 		if ner_dict[triple["subject"]] == 'PERSON':
	# 			print(triple["subject"] +" is a PERSON")
	#
	# 	if pos_dict[triple["object"].split(" ")[0]]=='NN' or pos_dict[triple["object"].split(" ")[0]]=='NNP':
	# 		print("object is a entitiy (instance of a class) ")
	# 		if ner_dict[triple["object"].split(" ")[0]] == 'LOCATION':
	# 			print(triple["object"] +" is a LOCATION")

# indentify_pos("400 victims in Dnkoluwaththa in Pitabaddra . Not enough food to eat. Very hard situation there")

# if 'nmod' in dependency_dict and dependency_dict['nmod'] in ner_dict:
# 	print("Entity: %s, ObjectAttr: %s, Value: %s" % (dependency_dict["nsubj"], dependency_dict["ROOT"], dependency_dict["nmod"]))
#
# 	generator = OwlGenerator("http://www.semanticweb.org/lily/ontologies/2016/10/disaster-management-system")
# 	entity1 = OwlEntity(ner_dict[dependency_dict["nsubj"]])
# 	entity2 = OwlEntity(ner_dict[dependency_dict["nmod"]])
# 	attr = OwlObjAttr(dependency_dict["ROOT"], entity1, entity2)
# 	generator.entities.append(entity1)
# 	generator.entities.append(entity2)
# 	generator.obj_attrs.append(attr)
# 	generator.write_owl("output_files/test2.owl")


# nlp = StanfordCoreNLP('http://localhost:9000')
# text = ('Lilan is a student')
# output = nlp.annotate(text,
# 	properties = {
# 		'annotators': 'ner',
# 		'outputFormat': 'json'
# 	})
# print(output['sentences'][0]['parse'])