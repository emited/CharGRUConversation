
import sys, getopt
import pandas as pd

'''Takes as input a subtitle processed csv file and returns
another csv with question answer couples associated with a batch number.
Within a batch, all question and answer sentences are the 
same length.
e.g: for batch k of size n, len(quest[i])==len(quest[j]) forall i, j
												and len(answ[i]) ==len(answ[j] ) forall i, j
'''

def chunks(l, n):
    return [l[i:i+n] for i in xrange(0, len(l), n) if len(l[i:i+n])==n]


def to_batch(df, batch_size=12):
	df['qlen'] = df.questions.apply(lambda x : len(x))
	df['alen'] = df.answers.apply(lambda x : len(x))
	grps = df.groupby(['qlen', 'alen'])
	index = grps.count().questions.index
	batches = []
	for i in index:
	    batches.append(chunks(grps.get_group(i)[['questions', 'answers']].values, batch_size))
	nbatches = [bi for b in batches for bi in b] #flatten
	nnbatches = [(i+1, bi[0], bi[1]) for i, b in enumerate(nbatches) for bi in b]
	return pd.DataFrame(nnbatches, columns=['nbatch', 'questions', 'answers'])

if __name__ == "__main__":

	opts, args = getopt.getopt(sys.argv[1:], 'i:o:b:')
	for o, a in opts:
		if o == '-i': #input file
			input_file = a
		else:
			input_file = 'qa.csv'
		if o == '-o': #output file
			output_file = 	a
			print('ok  ')
			print(output_file)
		else:
			output_file = 'make_batches_out.csv'
		if o == '-b': #batch size
			batch_size = int(a)
		else:
			batch_size = 12

	print('loading csv...')
	df = pd.read_csv(input_file)
	print('converting to batches...')
	ndf = to_batch(df, batch_size)
	print('saving to csv...')
	print(output_file	)
	ndf.to_csv(output_file, index=False)
	print('done!')
