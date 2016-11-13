
require 'csvigo'
local num2char = require 'num2char'
local char2num = require 'char2num'

local DataSetLoader = {}
DataSetLoader.__index = DataSetLoader


function DataSetLoader.create(filename, filetype, batchSize)
	local self = {}
	setmetatable(self, DataSetLoader)
	self.filename = filename
	if filetype == 'batch' then
		self:loadBatchFile(filename)
	else
		self.batchSize = batchSize
		self:loadFile(filename)
		self:createBatches(batchSize)
	end
	self.currentBatch = 0
	self.evaluatedBatches = 0
	return self
end

function DataSetLoader:loadBatchFile(filename)
	print('loading batch data file...')
	self.data = csvigo.load({path=filename, verbose=false, mode='tidy'})
	print('preparing batches...')
	self.data.size = #self.data.questions
	local k = 1
	local X, Y = {}, {}
	local q, a = {}, {}
	local stopChar = self.getAlphabetSize()
	for i = 1, self.data.size do
		if tonumber(self.data.nbatch[i]) > k then
			local nQChars = q[1]:size(1)
			local nAChars = a[1]:size(1)
			X[k] = torch.Tensor(#q, nQChars+1):fill(stopChar)
			Y[k] = torch.Tensor(#a, nAChars+1):fill(stopChar)
			for j=1, #q do
				X[k][j]:sub(1, nQChars):copy(q[j])
				Y[k][j]:sub(1, nAChars):copy(a[j])
			end
			q, a = {}, {}
			k = k + 1
		end
		q[#q+1] = self.string2tensor(self.data.questions[i])
		a[#a+1] = self.string2tensor(self.data.answers[i])
	end
	print(X)
	self.X = X
	self.Y = Y
	return self.X, self.Y
end

function DataSetLoader:loadFile(filename)
	print('loading data file...')
	self.data = csvigo.load({path=filename, verbose=false})
	assert(#self.data.questions == #self.data.answers)
	self.data.size = #self.data.questions
	return self.data
end

function DataSetLoader.getAlphabetSize()
	return #num2char+1 --add stop char
end

function DataSetLoader:createBatches(batchSize)
	print('creating batches...')
	local X, Y = {}, {}
	local k = 1
	for i=1, self.data.size, batchSize do
		if self.data.size - i < batchSize then break end
		local qmax, amax = 0, 0
		local q, a = {}, {}
		for j=1, batchSize do
			q[j] = self.string2tensor(self.data.questions[i+j+1])
			a[j] = self.string2tensor(self.data.answers[i+j+1])
			qmax = math.max(q[j]:size(1), qmax)
			amax = math.max(a[j]:size(1), amax)
		end
		local stopChar = self.getAlphabetSize()
		X[k] = torch.ByteTensor(batchSize, qmax+1):fill(stopChar) -- fill w/ stop char value
		Y[k] = torch.ByteTensor(batchSize, amax+1):fill(stopChar) -- fill w/ stop char value
		for j=1, batchSize do
			X[k][j]:sub(1, q[j]:size(1)):copy(q[j])
			Y[k][j]:sub(1, a[j]:size(1)):copy(a[j])
		end
		k = k + 1 
	end
	self.X = X
	self.Y = Y
	return self.X, self.Y
end

function DataSetLoader:nextBatch()
	self.currentBatch = (self.currentBatch % #self.X) + 1
	self.evaluatedBatches = self.evaluatedBatches + 1
	return self.X[self.currentBatch], self.Y[self.currentBatch]
end

function DataSetLoader.string2tensor(s)
	local t = torch.ByteTensor(#s)
	for i=1, #s do
		local num = char2num[s:sub(i, i)]
		t[i] = tonumber(num)
	end
	return t
end

function DataSetLoader.tensor2string(t)
	local s = {}
	for i=1, t:size(1)-1 do
		s[i] = num2char[t[i]]
	end
	return table.concat(s)
end

return DataSetLoader