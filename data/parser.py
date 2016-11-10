
import pysrt
import glob
import pandas as pd

mapping = {
	'a':1,
	'b':2,
	'c':3,
	'd':4,
	'e':5,
	'f':6,
	'g':7,
	'h':8,
	'i':9,
	'j':10,
	'k':11,
	'l':12,
	'm':13,
	'n':14,
	'o':15,
	'p':16,
	'q':17,
	'r':18,
	's':19,
	't':20,
	'u':21,
	'v':22,
	'w':23,
	'x':24,
	'y':25,
	'z':26,
	' ':27,
	'!':28,
	',':29,
	'?':30,
	';':31,
	'1':32,
	'2':33,
	'3':34,
	'4':35,
	'5':36,
	'6':37,
	'7':38,
	'8':39,
	'9':40,
	'.':41
}

inv_mapping = {v:k for k, v in mapping.iteritems()}

def preprocess(s):
	ns = s.lower()
	ns = ns.replace('<i>', '')
	ns = ns.replace('</i>', '')
	ns = ns.replace('\n', ' ')
	ns = ns.replace('-', ' ')
	ns = filter(lambda c: c in mapping.keys(), ns)
	ns = ns[:100] #taking only 99 first characters
	return ns

def char2num(s):
	return map(lambda c: mapping[c], s)

def num2char(n):
	return ''.join(map(lambda c:inv_mapping[c], n))

if __name__ == "__main__":
	results = []
	filenames = glob.glob('series/**/*.srt')
	for i, filename in enumerate(filenames):
		print 'parsing '+str(i)+' of '+str(len(filenames))+' files : '+filename
		subs = pysrt.open(filename, encoding='utf-8')
		text = [preprocess(s.text) for s in subs]
		if len(text)%2==1: text.pop()
		results += [(text[i], text[i+1]) for i in range(len(text)-1) if len(text[i])>2 and len(text[i+1])>2]
	pd.DataFrame(results).to_csv('question_answers.csv', header=['questions', 'answers'], index=False)