from nltk.corpus import wordnet as wn
import util
import sys

entities_dict = util.read_ontology_info("/home/lily/Projects/disaster-management-system/lilan/ontology_files/disaster-management-system-ontology.owl")

word = "family"
more_general_word = "entity"

for synset in wn.synsets(word, pos=wn.NOUN):
	for path in synset.hypernym_paths():
		for el in path:
			for lemma in el.lemmas():
				match = lemma.name()
				if more_general_word == match:
					print("Found a match for %s: %s" % (word, more_general_word))